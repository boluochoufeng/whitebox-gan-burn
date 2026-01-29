use burn::{
    Tensor,
    module::Module,
    nn::{
        PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::Backend,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use burn_import::pytorch::PyTorchFileRecorder;

#[derive(Module, Debug)]
pub struct VGG19<B: Backend> {
    // Block 1
    conv1_1: Conv2d<B>,
    conv1_2: Conv2d<B>,
    pool1: MaxPool2d,

    // Block 2
    conv2_1: Conv2d<B>,
    conv2_2: Conv2d<B>,
    pool2: MaxPool2d,

    // Block 3
    conv3_1: Conv2d<B>,
    conv3_2: Conv2d<B>,
    conv3_3: Conv2d<B>,
    conv3_4: Conv2d<B>,
    pool3: MaxPool2d,

    // Block 4
    conv4_1: Conv2d<B>,
    conv4_2: Conv2d<B>,
    conv4_3: Conv2d<B>,
    conv4_4: Conv2d<B>,
    // pool4: MaxPool2d,

    // Block 5
    // conv5_1: Conv2d<B>,
    // conv5_2: Conv2d<B>,
    // conv5_3: Conv2d<B>,
    // conv5_4: Conv2d<B>,
    // pool5: MaxPool2d,
    relu: Relu,
}

impl<B: Backend> VGG19<B> {
    pub fn new(device: &B::Device) -> Self {
        let pool_config = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]);
        Self {
            // Block 1
            conv1_1: Self::conv_layer(3, 64, device),
            conv1_2: Self::conv_layer(64, 64, device),
            pool1: pool_config.init(),

            // Block 2
            conv2_1: Self::conv_layer(64, 128, device),
            conv2_2: Self::conv_layer(128, 128, device),
            pool2: pool_config.init(),

            // Block 3
            conv3_1: Self::conv_layer(128, 256, device),
            conv3_2: Self::conv_layer(256, 256, device),
            conv3_3: Self::conv_layer(256, 256, device),
            conv3_4: Self::conv_layer(256, 256, device),
            pool3: pool_config.init(),

            // Block 4
            conv4_1: Self::conv_layer(256, 512, device),
            conv4_2: Self::conv_layer(512, 512, device),
            conv4_3: Self::conv_layer(512, 512, device),
            conv4_4: Self::conv_layer(512, 512, device),
            // pool4: pool_config.init(),

            // Block 5
            // conv5_1: Self::conv_layer(512, 512, device),
            // conv5_2: Self::conv_layer(512, 512, device),
            // conv5_3: Self::conv_layer(512, 512, device),
            // conv5_4: Self::conv_layer(512, 512, device),
            // pool5: pool_config.init(),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self.vgg_normalize(x);

        x = self.conv1_1.forward(x);
        x = self.relu.forward(x);
        x = self.conv1_2.forward(x);
        x = self.relu.forward(x);
        x = self.pool1.forward(x);

        x = self.conv2_1.forward(x);
        x = self.relu.forward(x);
        x = self.conv2_2.forward(x);
        x = self.relu.forward(x);
        x = self.pool2.forward(x);

        x = self.conv3_1.forward(x);
        x = self.relu.forward(x);
        x = self.conv3_2.forward(x);
        x = self.relu.forward(x);
        x = self.conv3_3.forward(x);
        x = self.relu.forward(x);
        x = self.conv3_4.forward(x);
        x = self.relu.forward(x);
        x = self.pool3.forward(x);

        x = self.conv4_1.forward(x);
        x = self.relu.forward(x);
        x = self.conv4_2.forward(x);
        x = self.relu.forward(x);
        x = self.conv4_3.forward(x);
        x = self.relu.forward(x);
        x = self.conv4_4.forward(x);
        x = self.relu.forward(x);

        x
    }

    fn vgg_normalize(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let device = x.device();
        let x = x * 0.5 + 0.5;
        let mean = Tensor::<B, 1>::from_data([0.485, 0.456, 0.406], &device).reshape([1, 3, 1, 1]);
        let std = Tensor::<B, 1>::from_data([0.229, 0.224, 0.225], &device).reshape([1, 3, 1, 1]);

        (x - mean) / std
    }

    fn conv_layer(in_channel: usize, out_channel: usize, device: &B::Device) -> Conv2d<B> {
        Conv2dConfig::new([in_channel, out_channel], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device)
    }
}

pub fn load_vgg_model<B: Backend>(
    weights_path: &str,
    device: &B::Device,
) -> Result<VGG19<B>, Box<dyn std::error::Error>> {
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .load(weights_path.into(), device)?;
    Ok(VGG19::new(device).load_record(record))
}

pub fn convert_model<B: Backend>(
    weights_path: &str,
    device: &B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("load vgg19 weights from pytorch: {}", weights_path);

    let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
    let record: VGG19Record<B> = recorder.load(weights_path.into(), device)?;

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    recorder.record(record, "vgg19.mpk".into())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::backend::Wgpu;

    use crate::vgg::{convert_model, load_vgg_model};

    type MyBackend = Wgpu<f32, i32>;

    #[test]
    fn test_import() {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let model = load_vgg_model::<MyBackend>("./vgg19.mpk", &device);
        assert!(model.is_ok(), "{:?}", model.err());
    }

    #[test]
    fn test_convert() {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let res = convert_model::<MyBackend>("./vgg19.pth", &device);
        assert!(res.is_ok(), "{:?}", res.err());
    }
}
