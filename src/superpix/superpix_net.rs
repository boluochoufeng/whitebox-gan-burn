use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        GroupNorm, GroupNormConfig, LeakyRelu, LeakyReluConfig, Tanh,
        conv::{Conv2d, Conv2dConfig},
        interpolate::Interpolate2d,
        pool::MaxPool2d,
    },
    prelude::Backend,
};

#[derive(Debug, Module)]
pub struct SuperpixNet<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    conv3: ConvBlock<B>,
    conv4: ConvBlock<B>,
    conv5: ConvBlock<B>,
    conv6: ConvBlock<B>,
    conv7: ConvBlock<B>,
    conv_out: Conv2d<B>,

    pool: MaxPool2d,
    tanh: Tanh,
    upsample: Interpolate2d,
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
        let conv1 = ConvBlockConfig::new([self.in_channel, self.channel])
            .with_kernel_size([7, 7])
            .with_stride([1, 1])
            .with_padding([3, 3])
            .init::<B>(device);

        todo!();
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
