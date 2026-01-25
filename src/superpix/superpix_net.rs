use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        GroupNorm, GroupNormConfig, LeakyRelu, LeakyReluConfig, PaddingConfig2d, Tanh,
        conv::{Conv2d, Conv2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig},
    },
    prelude::Backend,
};

#[derive(Debug, Module)]
pub struct SuperpixNet<B: Backend> {
    conv: ConvBlock<B>,

    conv1: ConvBlock<B>,
    down1: DownsamplingBlock<B>,

    conv2: ConvBlock<B>,
    down2: DownsamplingBlock<B>,

    conv3: ConvBlock<B>,
    down3: DownsamplingBlock<B>,

    conv4: ConvBlock<B>,
    up4: UpsamplingBlock<B>,

    conv5: ConvBlock<B>,
    up5: UpsamplingBlock<B>,

    conv6: ConvBlock<B>,
    up6: UpsamplingBlock<B>,

    conv_out: Conv2d<B>,

    tanh: Tanh,
}

impl<B: Backend> SuperpixNet<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x0 = self.conv.forward(x); // [256, 256]

        let x1 = self.down1.forward(x0.clone());
        let x1 = self.conv1.forward(x1); // [128, 128]

        let x2 = self.down2.forward(x1.clone());
        let x2 = self.conv2.forward(x2); // [64, 64]

        let x3 = self.down3.forward(x2.clone());
        let x3 = self.conv3.forward(x3); // [32, 32]

        let x4 = self.up4.forward(x3); // [64, 64]
        let x4 = self.conv4.forward(x4 + x2);

        let x5 = self.up5.forward(x4); // [128, 128]
        let x5 = self.conv5.forward(x5 + x1);

        let x6 = self.up6.forward(x5); // [256, 256]
        let x6 = self.conv6.forward(x6 + x0);

        let out = self.conv_out.forward(x6);
        self.tanh.forward(out)
    }
}

#[derive(Debug, Config)]
pub struct SuperpixNetConfig {
    #[config(default = "3")]
    in_channel: usize,
    #[config(default = "32")]
    channel: usize,
}

impl SuperpixNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SuperpixNet<B> {
        let conv = ConvBlockConfig::new([self.in_channel, self.channel])
            .with_kernel_size([7, 7])
            .with_stride([1, 1])
            .with_padding([3, 3])
            .init::<B>(device);
        let down1 = DownsamplingBlock::<B>::new([self.channel, self.channel], device);
        let conv1 = ConvBlockConfig::new([self.channel, self.channel * 2]).init::<B>(device);

        let down2 = DownsamplingBlock::<B>::new([self.channel * 2, self.channel * 2], device);
        let conv2 = ConvBlockConfig::new([self.channel * 2, self.channel * 4]).init::<B>(device);

        let down3 = DownsamplingBlock::<B>::new([self.channel * 4, self.channel * 4], device);
        let conv3 = ConvBlockConfig::new([self.channel * 4, self.channel * 8]).init::<B>(device);

        let up4 = UpsamplingBlock::<B>::new([self.channel * 8, self.channel * 4], device);
        let conv4 = ConvBlockConfig::new([self.channel * 4, self.channel * 4]).init::<B>(device);

        let up5 = UpsamplingBlock::<B>::new([self.channel * 4, self.channel * 2], device);
        let conv5 = ConvBlockConfig::new([self.channel * 2, self.channel * 2]).init::<B>(device);

        let up6 = UpsamplingBlock::<B>::new([self.channel * 2, self.channel], device);
        let conv6 = ConvBlockConfig::new([self.channel, self.channel]).init::<B>(device);

        let conv_out = Conv2dConfig::new([self.channel, self.in_channel], [7, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .init::<B>(device);

        let tanh = Tanh::new();

        SuperpixNet {
            conv,
            conv1,
            down1,
            conv2,
            down2,
            conv3,
            down3,
            conv4,
            up4,
            conv5,
            up5,
            conv6,
            up6,
            conv_out,
            tanh,
        }
    }
}

#[derive(Debug, Module)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: GroupNorm<B>,
    lrelu: LeakyRelu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self.conv.forward(x);
        x = self.norm.forward(x);
        self.lrelu.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ConvBlockConfig {
    channels: [usize; 2],
    #[config(default = "[3, 3]")]
    kernel_size: [usize; 2],
    #[config(default = "[1, 1]")]
    stride: [usize; 2],
    #[config(default = "[1, 1]")]
    padding: [usize; 2],
    #[config(default = "0.1")]
    negative_slope: f64,
}

impl ConvBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBlock<B> {
        ConvBlock {
            conv: Conv2dConfig::new(self.channels, self.kernel_size)
                .with_stride(self.stride)
                .with_padding(burn::nn::PaddingConfig2d::Explicit(
                    self.padding[0],
                    self.padding[1],
                ))
                .init(device),
            norm: GroupNormConfig::new(1, self.channels[1]).init(device),
            lrelu: LeakyReluConfig::new()
                .with_negative_slope(self.negative_slope)
                .init(),
        }
    }
}

#[derive(Debug, Module)]
pub struct DownsamplingBlock<B: Backend> {
    conv: Conv2d<B>,
    lrelu: LeakyRelu,
}

impl<B: Backend> DownsamplingBlock<B> {
    pub fn new(channels: [usize; 2], device: &B::Device) -> Self {
        Self {
            conv: Conv2dConfig::new(channels, [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            lrelu: LeakyReluConfig::new().with_negative_slope(0.1).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        self.lrelu.forward(x)
    }
}

#[derive(Debug, Module)]
pub struct UpsamplingBlock<B: Backend> {
    conv: Conv2d<B>,
    lrelu: LeakyRelu,
    upsample: Interpolate2d,
}

impl<B: Backend> UpsamplingBlock<B> {
    pub fn new(channels: [usize; 2], device: &B::Device) -> Self {
        Self {
            conv: Conv2dConfig::new(channels, [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            lrelu: LeakyReluConfig::new().with_negative_slope(0.1).init(),
            upsample: Interpolate2dConfig::new()
                .with_scale_factor(Some([2.0, 2.0]))
                .with_mode(burn::nn::interpolate::InterpolateMode::Cubic)
                .init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.lrelu.forward(x);
        self.upsample.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        backend::{Autodiff, Wgpu},
    };

    use crate::superpix::superpix_net::SuperpixNetConfig;
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    fn device() -> <MyBackend as burn::prelude::Backend>::Device {
        Default::default()
    }

    #[test]
    fn forward_shape_and_finite_wgpu() {
        let device = device();

        let unet = SuperpixNetConfig::new().init::<MyAutodiffBackend>(&device);
        let input: Tensor<MyAutodiffBackend, 4> = Tensor::zeros([1, 3, 256, 256], &device);
        let output = unet.forward(input);

        assert_eq!([1, 3, 256, 256], output.dims());

        println!("{}", unet);
    }
}
