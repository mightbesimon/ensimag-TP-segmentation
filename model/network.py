# import the necessary packages
from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
import torch


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))


class SimpleConv(Module):
    def __init__(self, enc_channels=(3, 16, 32, 64),
                 dec_channels=(64, 32, 16, 1)):
        super().__init__()

        # store the encoder blocks and maxpooling layer
        self.enc_blocks = ModuleList([Block(enc_channels[i], enc_channels[i+1])
                                     for i in range(len(enc_channels) - 1)])
        # TODO: prepare pooling

        # initialize decoder # channels, upsampler blocks, and decoder blocks
        self.channels = dec_channels
        self.dec_blocks = ModuleList(
            [Block(dec_channels[i], dec_channels[i + 1])
             for i in range(len(dec_channels) - 1)])
        # TODO: prepare upconvolutions

    def forward(self, x):
        # loop through the encoder blocks
        for block in self.enc_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            # TODO: pooling

        # decoder: loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            # TODO: upconvolutions
            x = self.dec_blocks[i](x)
        #
        # return the final decoder output
        return x


def make_blocks(channels, block_func=lambda i, ch: Block(ch[i], ch[i+1])):
    return ModuleList([
        block_func(i, channels)
        for i in range(len(channels) - 1)
    ])


class HourGlass(Module):
    def __init__(self,
            enc_channels=( 3, 16, 32, 64),
            dec_channels=(64, 32, 16,  1)):

        super().__init__()

        # store the encoder blocks and maxpooling layer
        self.enc_blocks = make_blocks(enc_channels)

        # prepare pooling: halves the spatial size after each encoder block
        self.pool = MaxPool2d(2)

        # dec_channels' last entry is the number of output classes, produced
        # by a standalone 1x1 conv after the last decoder block, not by the
        # block itself; the block keeps halving channels like the others,
        # so it ends at half of the previous (second-to-last) channel count
        block_channels = dec_channels[:-1] + (dec_channels[-2] // 2,)
        self.channels = dec_channels
        self.dec_blocks = make_blocks(block_channels)

        # prepare upconvolutions: doubles the spatial size back up, one per
        # decoder block, so total encoder pools == total decoder upconvs
        self.upconvs = make_blocks(block_channels,
            block_func=lambda i, ch: ConvTranspose2d(ch[i], ch[i], 2, 2))

        # final 1x1 conv mapping decoder features to the segmentation mask
        self.head = Conv2d(block_channels[-1], dec_channels[-1], 1)

    def forward(self, x):
        # loop through the encoder blocks
        # pass the inputs through the current encoder block, then
        # apply maxpooling on the output to shrink the feature map
        for block in self.enc_blocks:
            x = block(x)
            x = self.pool(x)

        # decoder: loop through the decoder blocks
        # upscale the features back up before the decoder block
        for upconv, dec_block in zip(self.upconvs, self.dec_blocks):
            x = upconv(x)
            x = dec_block(x)

        # map decoder features to the segmentation mask
        return self.head(x)


class UNet(Module):

    ...
