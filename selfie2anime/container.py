from selfie2anime.cyclegan import Discriminator, VanillaGenerator

import argparse
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_weights(model: nn.Module, mean: float, std: float) -> None:
    r"""
    Initialize model weights from a Gaussian distribution $\mathcal{N}(\mu, \sigma)$
    """

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, mean=mean, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0.0)

        if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(module.weight.data, mean=1.0, std=std)
            nn.init.constant_(module.bias.data, 0.0)


class CycleGAN:
    """
    A container handling cyclegan generators and discriminators.
    This implement currently only supports Vanilla CycleGAN model proposed originally in -
    "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
    with future plans on implementing the U-GAT-IT varient as well proposed in -
    "U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation"
    """

    def __init__(self, config: argparse.Namespace, mode: str = "vanilla") -> None:
        # This implementation uses A to define selfie and B to define Anime

        if mode == "vanilla":
            self.gen_A = VanillaGenerator(
                in_channels=config.in_channels,
                num_features=config.num_features,
                num_residuals=config.num_residuals,
            ).to(DEVICE)

            self.gen_B = VanillaGenerator(
                in_channels=config.in_channels,
                num_features=config.num_features,
                num_residuals=config.num_residuals,
            ).to(DEVICE)

        # use same discriminator architectures for all modes
        self.disc_A = Discriminator(
            in_channels=config.in_channels,
            disc_channels=config.disc_channels,
        ).to(DEVICE)

        self.disc_B = Discriminator(
            in_channels=config.in_channels,
            disc_channels=config.disc_channels,
        )

        # weight initialization
        initialize_weights(self.gen_A, mean=config.init_mean, std=config.init_std)
        initialize_weights(self.gen_B, mean=config.init_mean, std=config.init_std)
        initialize_weights(self.disc_A, mean=config.init_mean, std=config.init_std)
        initialize_weights(self.disc_B, mean=config.init_mean, std=config.init_std)
