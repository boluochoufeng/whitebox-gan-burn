
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        LeakyRelu, LeakyReluConfig, PaddingConfig2d, Tanh,
        conv::{Conv2d, Conv2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig},
    },
    prelude::Backend,
};

#[derive(Debug, Module)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    lrelu: LeakyRelu,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(input.clone());
        let x = self.lrelu.forward(x);
        let x = self.conv2.forward(x);
        x + input
    }
}

#[derive(Config, Debug)]
pub struct ResidualBlockConfig {
    channels: [usize; 2],
    kernel_size: [usize; 2],
    #[config(default = "[1, 1]")]
    stride: [usize; 2],
    #[config(default = "PaddingConfig2d::Same")]
    padding: PaddingConfig2d,
    #[config(default = "0.2")]
    negative_slop: f64,
}

impl ResidualBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualBlock<B> {
        ResidualBlock {
            conv1: Conv2dConfig::new(self.channels, self.kernel_size)
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new(self.channels, self.kernel_size)
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            lrelu: LeakyReluConfig::new()
                .with_negative_slope(self.negative_slop)
                .init(),
        }
    }
}

#[derive(Debug, Module)]
pub struct Generator<B: Backend> {
    conv: Conv2d<B>,
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    res_blocks: Vec<ResidualBlock<B>>,
    conv5: Conv2d<B>,
    conv6: Conv2d<B>,
    conv7: Conv2d<B>,
    conv8: Conv2d<B>,
    conv9: Conv2d<B>,
    lrelu: LeakyRelu,
    upsample: Interpolate2d,
    act: Tanh,
}

impl<B: Backend> Generator<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x0 = self.conv.forward(input);
        x0 = self.lrelu.forward(x0); // (256, 256, 32)

        let mut x1 = self.conv1.forward(x0.clone());
        x1 = self.lrelu.forward(x1);
        x1 = self.conv2.forward(x1);
        x1 = self.lrelu.forward(x1); // (128, 128, 64)

        let mut x2 = self.conv3.forward(x1.clone());
        x2 = self.lrelu.forward(x2);
        x2 = self.conv4.forward(x2);
        x2 = self.lrelu.forward(x2); // (64, 64, 128)

        let mut x3 = x2;
        for block in self.res_blocks.iter() {
            x3 = block.forward(x3); // (64, 64, 128)
        }

        let mut x4 = self.conv5.forward(x3);
        x4 = self.lrelu.forward(x4); // (64, 64, 64)

        let mut x5 = self.upsample.forward(x4);
        x5 = self.conv6.forward(x5 + x1);
        x5 = self.lrelu.forward(x5);
        x5 = self.conv7.forward(x5);
        x5 = self.lrelu.forward(x5); //(128, 128, 32)

        let mut x6 = self.upsample.forward(x5);
        x6 = self.conv8.forward(x6 + x0);
        x6 = self.lrelu.forward(x6);
        x6 = self.conv9.forward(x6);

        self.act.forward(x6)
    }
}

#[derive(Config, Debug)]
pub struct GeneratorConfig {
    #[config(default = "3")]
    input_channel: usize,
    #[config(default = "32")]
    base_channel: usize,
    #[config(default = "4")]
    num_blocks: usize,
    #[config(default = "0.2")]
    negative_slope: f64,
}

impl GeneratorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Generator<B> {
        let input_channel = self.input_channel;
        let channel = self.base_channel;

        let conv = Self::conv_block(input_channel, channel, [7, 7], [1, 1], 3, device);
        let conv1 = Self::conv_block(channel, channel, [3, 3], [2, 2], 1, device);
        let conv2 = Self::conv_block(channel, channel * 2, [3, 3], [1, 1], 1, device);
        let conv3 = Self::conv_block(channel * 2, channel * 2, [3, 3], [2, 2], 1, device);
        let conv4 = Self::conv_block(channel * 2, channel * 4, [3, 3], [1, 1], 1, device);

        let mut res_blocks = Vec::with_capacity(self.num_blocks);
        for _ in 0..self.num_blocks {
            res_blocks
                .push(ResidualBlockConfig::new([channel * 4, channel * 4], [3, 3]).init(device));
        }

        let conv5 = Self::conv_block(channel * 4, channel * 2, [3, 3], [1, 1], 1, device);
        let conv6 = Self::conv_block(channel * 2, channel * 2, [3, 3], [1, 1], 1, device);
        let conv7 = Self::conv_block(channel * 2, channel, [3, 3], [1, 1], 1, device);
        let conv8 = Self::conv_block(channel, channel, [3, 3], [1, 1], 1, device);
        let conv9 = Self::conv_block(channel, 3, [7, 7], [1, 1], 3, device);

        let lrelu = LeakyReluConfig::new()
            .with_negative_slope(self.negative_slope)
            .init();
        let upsample = Interpolate2dConfig::new()
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest) // 原论文是双线性插值
            .with_scale_factor(Some([2.0, 2.0]))
            .init();
        let act = Tanh::new();

        Generator {
            conv,
            conv1,
            conv2,
            conv3,
            conv4,
            res_blocks,
            conv5,
            conv6,
            conv7,
            conv8,
            conv9,
            lrelu,
            upsample,
            act,
        }
    }

    fn conv_block<B: Backend>(
        in_channel: usize,
        out_channel: usize,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: usize,
        device: &B::Device,
    ) -> Conv2d<B> {
        Conv2dConfig::new([in_channel, out_channel], kernel_size)
            .with_stride(stride)
            .with_padding(PaddingConfig2d::Explicit(padding, padding))
            .init(device)
    }
}
