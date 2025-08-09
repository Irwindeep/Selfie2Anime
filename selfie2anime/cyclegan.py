from typing import List
from selfie2anime.utils.conv import ConvBlock, UpConvBlock

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Implementation of discriminator as a PatchGAN.
    """

    def __init__(
        self,
        in_channels: int,
        disc_channels: List[int],
    ) -> None:
        super(Discriminator, self).__init__()

        # Initial layer with no normalization
        self.initial = ConvBlock(
            in_channels=in_channels,
            out_channels=disc_channels[0],
            kernel_size=4,
            stride=2,
            padding=1,
            activation="leaky_relu",
            norm="none",
        )

        # discriminator block with ConvBlock
        in_channs, out_channs = disc_channels[:-1], disc_channels[1:]
        self.disc = nn.Sequential(
            *[
                ConvBlock(
                    in_channels=in_chann,
                    out_channels=out_chann,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    activation="leaky_relu",
                    norm="instance_norm",
                )
                for in_chann, out_chann in zip(in_channs, out_channs)
            ],
        )

        # final layer to get single channeled output
        self.final = nn.Conv2d(
            in_channels=out_channs[-1],
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.initial(input)
        output = self.disc(output)
        output = self.final(output)

        return output


class ResidualBlock(nn.Module):
    """
    A residual block using 2 Convolutional Blocks.
    """

    def __init__(self, channels: int) -> None:
        super(ResidualBlock, self).__init__()

        conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            norm="instance_norm",
        )
        conv2 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            activation="none",
            norm="instance_norm",
        )

        self.block = nn.Sequential(conv1, conv2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.block(input)


class VanillaGenerator(nn.Module):
    """
    Implementation of Vanilla CycleGAN resnet based generator discribed in -
    `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`
    """

    def __init__(self, in_channels: int, num_features: int, num_residuals: int) -> None:
        super(VanillaGenerator, self).__init__()

        self.initial = ConvBlock(
            in_channels=in_channels,
            out_channels=num_features,
            kernel_size=7,
            padding=3,
            padding_mode="reflect",
            norm="instance_norm",
        )

        self.down_proj = nn.Sequential(
            *[
                ConvBlock(
                    in_channels=num_features * (2**i),
                    out_channels=num_features * (2 ** (i + 1)),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    padding_mode="reflect",
                    norm="instance_norm",
                )
                for i in range(2)
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels=num_features * 4) for _ in range(num_residuals)]
        )

        self.up_proj = nn.Sequential(
            *[
                UpConvBlock(
                    in_channels=num_features * (2 ** (i + 1)),
                    out_channels=num_features * (2**i),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    norm="instance_norm",
                )
                for i in range(1, -1, -1)
            ]
        )

        self.final = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features,
                out_channels=in_channels,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            ),
            nn.Tanh(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.initial(input)
        output = self.down_proj(output)
        output = self.residual_blocks(output)
        output = self.up_proj(output)
        output = self.final(output)

        return output
