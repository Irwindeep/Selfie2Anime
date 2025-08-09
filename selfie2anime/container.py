from typing import Dict, Tuple
from selfie2anime.cyclegan import Discriminator, VanillaGenerator
from selfie2anime.utils.loss import _ImgTuple, DiscLoss, CycleGANLossGen
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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

        if isinstance(module, (nn.BatchNorm2d)):
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
        self.config = config

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

            self.gen_loss_fn = CycleGANLossGen(
                lmbd_cyc=config.lmbd_cyc, lmbd_iden=config.lmbd_iden
            )

        # use same discriminator architectures for all modes
        self.disc_A = Discriminator(
            in_channels=config.in_channels,
            disc_channels=config.disc_channels,
        ).to(DEVICE)

        self.disc_B = Discriminator(
            in_channels=config.in_channels,
            disc_channels=config.disc_channels,
        )
        self.disc_loss_fn = DiscLoss()

        # weight initialization
        initialize_weights(self.gen_A, mean=config.init_mean, std=config.init_std)
        initialize_weights(self.gen_B, mean=config.init_mean, std=config.init_std)
        initialize_weights(self.disc_A, mean=config.init_mean, std=config.init_std)
        initialize_weights(self.disc_B, mean=config.init_mean, std=config.init_std)

        self.disc_train_losses = {}
        self.gen_train_losses = {}

    def train(
        self,
        train_loader: DataLoader[_ImgTuple],
        optim_disc: torch.optim.Optimizer,
        optim_gen: torch.optim.Optimizer,
    ) -> None:
        for epoch in range(1, self.config.num_epochs + 1):
            n = len(str(self.config.num_epochs))
            desc = f"Epoch [{epoch:0{n}d}/{self.config.num_epochs}]"
            num_batches = len(train_loader)
            iter = tqdm(train_loader, desc=desc, leave=False)

            disc_losses, gen_losses = self._train_epoch(iter, optim_disc, optim_gen)

            for key, val in disc_losses.items():
                if key not in self.disc_train_losses:
                    self.disc_train_losses[key] = []
                self.disc_train_losses[key].append(val / num_batches)

            for key, val in gen_losses.items():
                if key not in self.gen_train_losses:
                    self.gen_train_losses[key] = []
                self.gen_train_losses[key].append(val / num_batches)

    def _train_epoch(
        self,
        iter: tqdm,
        optim_disc: torch.optim.Optimizer,
        optim_gen: torch.optim.Optimizer,
    ) -> Tuple[Dict, ...]:
        disc_losses, gen_losses = {}, {}

        for real_A, real_B in iter:
            real_A = real_A.to(DEVICE)
            real_B = real_B.to(DEVICE)

            fake_A = self.gen_A(real_B)
            fake_B = self.gen_B(real_A)

            ## Compute Discriminator Loss
            disc_real_A = self.disc_A(real_A)
            disc_real_B = self.disc_B(real_B)

            disc_fake_A = self.disc_A(fake_A.detach())
            disc_fake_B = self.disc_B(fake_B.detach())

            disc_loss = self.disc_loss_fn(
                disc_real=(disc_real_A, disc_real_B),
                disc_fake=(disc_fake_A, disc_fake_B),
            )

            ## Compute Generator Loss
            disc_fake_A = self.disc_A(fake_A)
            disc_fake_B = self.disc_B(fake_B)

            cyc_A = self.gen_A(fake_B)
            cyc_B = self.gen_B(fake_A)

            iden_imgs = None
            if self.config.lmbd_iden > 0.0:
                iden_A = self.gen_A(real_A)
                iden_B = self.gen_B(real_B)

                iden_imgs = (iden_A, iden_B)

            gen_loss = self.gen_loss_fn(
                real_imgs=(real_A, real_B),
                cyc_imgs=(cyc_A, cyc_B),
                disc_fake=(disc_fake_A, disc_fake_B),
                iden_imgs=iden_imgs,
            )

            ## Train Discriminators
            optim_disc.zero_grad()
            disc_loss["loss_disc"].backward()
            optim_disc.step()

            ## Train Generators
            optim_gen.zero_grad()
            gen_loss["loss_gen"].backward()
            optim_gen.step()

            iter.set_postfix(
                {
                    "Disc Loss": f"{disc_loss['loss_disc'].item():.4f}",
                    "Gen Loss": f"{gen_loss['loss_gen'].item():.4f}",
                }
            )

            for key, val in disc_loss.items():
                disc_losses[key] = val.item() + disc_losses.get(key, 0.0)

            for key, val in gen_loss.items():
                gen_losses[key] = val.item() + gen_losses.get(key, 0.0)

        return disc_losses, gen_losses
