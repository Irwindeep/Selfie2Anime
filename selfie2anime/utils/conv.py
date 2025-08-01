import torch
import torch.nn as nn

from typing import Union
from torch.nn.utils import spectral_norm


class ConvBlock(nn.Module):
    """
    Generalized Convolution-Normalization-Activation block.
    Activations: ReLU, LeakyReLU.
    Normalizations: Batch, Instance, Spectral normalizations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = "same",
        padding_mode: str = "zeros",
        activation: str = "relu",
        norm: str = "batch_norm",
    ) -> None:
        super().__init__()

        act_fn = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.2),
        }.get(activation, nn.ReLU())

        # default normalization is Identity to handle spectral normalization
        norm_fn = {
            "batch_norm": nn.BatchNorm2d(num_features=out_channels),
            "instance_norm": nn.InstanceNorm2d(num_features=out_channels),
        }.get(norm, nn.Identity())

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
        )
        if norm == "spectral_norm":
            # apply spectral normalization on conv weights
            conv = spectral_norm(conv)

        self.block = nn.Sequential(conv, norm_fn, act_fn)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.block(input)


class UpConvBlock(nn.Module):
    """
    Generalized Transposed Convolution-Normalization-Activation block.
    Activations: ReLU, LeakyReLU.
    Normalizations: Batch, Instance, Spectral Normalizations.

    NOTE: Original CycleGAN paper suggests the use of a fractionally-strided convolution
    for this block which can be implemented in two ways:
    >>> UpSample + Conv
    >>> Transposed Convolution

    Transposed Convolution is chosen in this implementation as it is `end-to-end` trainable.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        activation: str = "relu",
        norm: str = "batch_norm",
    ) -> None:
        super().__init__()

        act_fn = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.2),
        }.get(activation, nn.ReLU())

        # default normalization is Identity to handle spectral normalization
        norm_fn = {
            "batch_norm": nn.BatchNorm2d(num_features=out_channels),
            "instance_norm": nn.InstanceNorm2d(num_features=out_channels),
        }.get(norm, nn.Identity())

        transpose_conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        if norm == "spectral_norm":
            # apply spectral normalization on transpose conv weights
            transpose_conv = spectral_norm(transpose_conv)

        self.block = nn.Sequential(transpose_conv, norm_fn, act_fn)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.block(input)
