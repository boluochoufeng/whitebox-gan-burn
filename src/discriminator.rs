use burn::{
    Tensor,
    config::Config,
    module::{Initializer, Module, Param},
    nn::{LeakyRelu, LeakyReluConfig},
    prelude::Backend,
    tensor::{Distribution, module::conv2d, ops::ConvOptions},
};

#[derive(Debug, Module)]
pub struct SNStateUV<B: Backend> {
    pub u: Param<Tensor<B, 2>>,
    pub power_iterations: usize,
}

impl<B: Backend> SNStateUV<B> {
    pub fn update_u_v(&mut self, w: Tensor<B, 4>) -> Option<Tensor<B, 2>> {
        let w_shape = w.dims();
        let height = w_shape[0];
        let w = w.reshape([height, w_shape[1] * w_shape[2] * w_shape[3]]);

        let mut u_hat = self.u.val();
        let mut v_hat = None;
        for _ in 0..self.power_iterations {
            let v_ = w.clone().transpose().matmul(u_hat);
            v_hat = Some(l2normalize(v_));

            let u_ = w.clone().matmul(v_hat.clone().unwrap());
            u_hat = l2normalize(u_);
        }

        if v_hat.is_none() {
            return None;
        }

        let u_hat = u_hat.detach();
        let v_hat = v_hat.unwrap().detach();

        let sigma = u_hat.clone().transpose().matmul(w.matmul(v_hat));
        self.u = Param::from_tensor(u_hat);

        Some(sigma)
    }
}

#[derive(Debug, Module)]
pub struct SNConv2d<B: Backend> {
    pub weight: Param<Tensor<B, 4>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub stride: [usize; 2],
    pub kernel_size: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
    pub padding: [usize; 2],
}

impl<B: Backend> SNConv2d<B> {
    pub fn forward(&self, input: Tensor<B, 4>, sn_state: &mut SNStateUV<B>) -> Tensor<B, 4> {
        let w_shape = self.weight.dims();
        if let Some(sigma) = sn_state.update_u_v(self.weight.val()) {
            conv2d(
                input,
                self.weight.val() / sigma.expand(w_shape),
                self.bias.as_ref().map(|bias| bias.val()),
                ConvOptions::new(self.stride, self.padding, self.dilation, self.groups),
            )
        } else {
            conv2d(
                input,
                self.weight.val(),
                self.bias.as_ref().map(|bias| bias.val()),
                ConvOptions::new(self.stride, self.padding, self.dilation, self.groups),
            )
        }
    }
}

#[derive(Config, Debug)]
pub struct SNConv2dConfig {
    pub channels: [usize; 2],
    pub kernel_size: [usize; 2],
    #[config(default = "[1, 1]")]
    pub stride: [usize; 2],
    #[config(default = "[0, 0]")]
    pub padding: [usize; 2],
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
    )]
    pub initializer: Initializer,
    #[config(default = true)]
    pub bias: bool,
    #[config(default = "1")]
    pub power_iterations: usize,
}

impl SNConv2dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> (SNConv2d<B>, SNStateUV<B>) {
        let groups = 1;
        let shape = [
            self.channels[1],
            self.channels[0],
            self.kernel_size[0],
            self.kernel_size[1],
        ];
        let k = self.kernel_size.iter().product::<usize>();
        let fan_in = self.channels[0] / groups * k;
        let fan_out = self.channels[1] / groups * k;

        let weight = self
            .initializer
            .init_with(shape, Some(fan_in), Some(fan_out), device);
        let mut bias = None;
        if self.bias {
            bias = Some(self.initializer.init_with(
                [self.channels[1]],
                Some(fan_in),
                Some(fan_out),
                device,
            ));
        }

        let height = weight.dims()[0];
        let u = Tensor::<B, 2>::random([height, 1], Distribution::Normal(0.0, 1.0), device);
        let u = l2normalize(u);
        let u = Param::from_tensor(u).set_require_grad(false);

        (
            SNConv2d {
                weight,
                bias,
                stride: self.stride,
                kernel_size: self.kernel_size,
                dilation: [1, 1],
                groups: groups,
                padding: self.padding,
            },
            SNStateUV {
                u,
                power_iterations: self.power_iterations,
            },
        )
    }
}

#[derive(Debug, Module)]
pub struct SNConv2dBlock<B: Backend> {
    conv: SNConv2d<B>,
    lrelu: LeakyRelu,
}

impl<B: Backend> SNConv2dBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>, st: &mut SNStateUV<B>) -> Tensor<B, 4> {
        let x = self.conv.forward(x, st);
        self.lrelu.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct SNConv2dBlockConfig {
    channel: [usize; 2],
    #[config(default = "[3, 3]")]
    kernel_size: [usize; 2],
    #[config(default = "[1, 1]")]
    stride: [usize; 2],
    #[config(default = "[1, 1]")]
    padding: [usize; 2],
    #[config(default = "0.2")]
    negative_slope: f64,
}

impl SNConv2dBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> (SNConv2dBlock<B>, SNStateUV<B>) {
        let (conv, st) = SNConv2dConfig::new(self.channel, self.kernel_size)
            .with_stride(self.stride)
            .with_padding(self.padding)
            .init::<B>(&device);
        (
            SNConv2dBlock {
                conv,
                lrelu: LeakyReluConfig::new()
                    .with_negative_slope(self.negative_slope)
                    .init(),
            },
            st,
        )
    }
}

#[derive(Debug, Module)]
pub struct Discriminator<B: Backend> {
    body: Vec<SNConv2dBlock<B>>,
    head: SNConv2d<B>,
}

impl<B: Backend> Discriminator<B> {
    pub fn forward(&self, x: Tensor<B, 4>, sts: &mut Vec<SNStateUV<B>>) -> Tensor<B, 4> {
        let mut x = x;
        for (conv, st) in self.body.iter().zip(sts.iter_mut()) {
            x = conv.forward(x, st);
        }
        x = self.head.forward(x, sts.last_mut().unwrap());

        x
    }
}

#[derive(Config, Debug)]
pub struct DiscriminatorConfig {
    #[config(default = "3")]
    in_channel: usize,
    #[config(default = "32")]
    base_channel: usize,
    #[config(default = "3")]
    num_blocks: usize,
    #[config(default = "0.2")]
    negative_slope: f64,
}

impl DiscriminatorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Discriminator<B>, Vec<SNStateUV<B>>) {
        let mut body = Vec::new();
        let mut sts = Vec::new();
        let channel = self.base_channel;
        let mut in_channel = self.in_channel;
        for i in 0..self.num_blocks {
            let (sn_conv, st) =
                SNConv2dBlockConfig::new([in_channel, channel * (2u32.pow(i as u32) as usize)])
                    .with_kernel_size([3, 3])
                    .with_stride([2, 2])
                    .with_padding([1, 1])
                    .init::<B>(device);
            body.push(sn_conv);
            sts.push(st);

            in_channel = channel * (2u32.pow(i as u32) as usize);

            let (sn_conv, st) =
                SNConv2dBlockConfig::new([in_channel, channel * (2u32.pow(i as u32) as usize)])
                    .with_kernel_size([3, 3])
                    .with_stride([1, 1])
                    .with_padding([1, 1])
                    .init::<B>(device);
            body.push(sn_conv);
            sts.push(st);

            in_channel = channel * 2u32.pow(i as u32) as usize;
        }

        let (head, st) = SNConv2dConfig::new([in_channel, 1], [1, 1])
            .with_padding([0, 0])
            .with_stride([1, 1])
            .init::<B>(device);
        sts.push(st);

        (Discriminator { body, head }, sts)
    }
}

fn l2normalize<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    const EPSILON: f32 = 1e-12;
    let norm = x.clone().powf_scalar(2.0).sum().sqrt();
    x / (norm.unsqueeze_dim(1) + EPSILON)
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        backend::{Autodiff, Wgpu},
        tensor::Distribution,
    };
    use nalgebra::DMatrix;

    use crate::discriminator::SNConv2dConfig;

    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    fn device() -> <MyBackend as burn::prelude::Backend>::Device {
        Default::default()
    }

    fn to_vec_f32_1d(data: burn::tensor::TensorData) -> Vec<f32> {
        data.into_vec().unwrap()
    }

    fn conv_cfg() -> SNConv2dConfig {
        SNConv2dConfig::new([3, 8], [3, 3])
    }

    #[test]
    fn forward_shape_and_finite_wgpu() {
        let device = device();
        let (sn, mut st) = conv_cfg().with_padding([1, 1]).init::<MyBackend>(&device);
        let x =
            Tensor::<MyBackend, 4>::random([4, 3, 28, 28], Distribution::Normal(0.0, 1.0), &device);

        let y = sn.forward(x, &mut st);
        assert_eq!(y.dims(), [4, 8, 28, 28]);

        let yv = to_vec_f32_1d(y.to_data());
        assert!(yv.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn u_updates_and_is_unit_norm() {
        let device = device();
        let (sn, mut st) = conv_cfg().with_padding([1, 1]).init::<MyBackend>(&device);

        let w = sn.weight.val();
        let u0 = to_vec_f32_1d(st.u.val().to_data());

        for _ in 0..10 {
            let _ = st.update_u_v(w.clone());
        }

        let u1_t = st.u.val();
        let u1 = to_vec_f32_1d(u1_t.to_data());

        let diff: f32 = u0.iter().zip(u1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0);

        let norm = u1.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "u norm={norm}");
    }

    #[test]
    fn spectral_norm_of_normalized_weight_close_to_1() {
        let device = device();
        let (sn, mut st) = conv_cfg()
            .with_power_iterations(25)
            .init::<MyBackend>(&device);

        let w = sn.weight.val();
        let [o, ic, kh, kw] = w.dims();
        let width = ic * kh * kw;

        let sigma = st.update_u_v(w.clone()).expect("sigma should exist");
        let w_sn = w / sigma.expand([o, ic, kh, kw]);

        let w_sn_host = to_vec_f32_1d(w_sn.to_data());

        let mat = DMatrix::<f32>::from_row_slice(o, width, &w_sn_host);

        let svd = mat.svd(true, true);
        let smax = svd.singular_values[0];

        assert!(
            (smax - 1.0).abs() < 1e-2,
            "max singular value of W_sn should be ~1, got {smax}"
        );
    }

    #[test]
    fn backward_runs_and_weight_has_grad() {
        let device = device();
        let (sn, mut st) = conv_cfg().init::<MyAutodiffBackend>(&device);

        let x = Tensor::<MyAutodiffBackend, 4>::random(
            [2, 3, 16, 16],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let y = sn.forward(x, &mut st);
        let loss = y.clone().powf_scalar(2.0).mean();

        let grads = loss.backward();

        let w = sn.weight.val();
        let grad_w = w.grad(&grads).expect("weight grad should exist");

        let gv = to_vec_f32_1d(grad_w.to_data());
        let sum_abs: f32 = gv.iter().map(|v| v.abs()).sum();
        assert!(sum_abs > 0.0, "grad seems zero");
    }
}
