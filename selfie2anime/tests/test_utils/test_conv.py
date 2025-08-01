import unittest
import torch

from selfie2anime.utils.conv import ConvBlock, UpConvBlock

conv_block = ConvBlock(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    activation="leaky_relu",
    norm="spectral_norm",
)

upconv_block = UpConvBlock(
    in_channels=16,
    out_channels=3,
    kernel_size=4,
    stride=2,
    padding=1,
    activation="leaky_relu",
)


class TestConvBlock(unittest.TestCase):
    def test_output_dim(self):
        x = torch.randn(1, 3, 16, 16)
        output = conv_block(x)

        # expected output size = (1, 16, 16, 16)
        assert output.size() == (1, 16, 16, 16)

    def test_spectral_norm(self):
        for module in conv_block.modules():
            if isinstance(module, torch.nn.Conv2d):
                assert hasattr(module, "weight_u")

    def test_norm_module(self):
        assert any(isinstance(m, torch.nn.Identity) for m in conv_block.modules())


class TestUpConvBlock(unittest.TestCase):
    def test_output_dim(self):
        x = torch.randn(1, 16, 8, 8)
        output = upconv_block(x)

        # expected output size = (1, 3, 16, 16)
        assert output.size() == (1, 3, 16, 16)

    def test_spectral_norm(self):
        for module in upconv_block.modules():
            if isinstance(module, torch.nn.Conv2d):
                assert not hasattr(module, "weight_u")

    def test_norm_module(self):
        assert any(isinstance(m, torch.nn.BatchNorm2d) for m in upconv_block.modules())
