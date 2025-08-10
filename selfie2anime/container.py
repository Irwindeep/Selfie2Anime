import os
from typing import Dict, Tuple

from selfie2anime.cyclegan import Discriminator, VanillaGenerator
from selfie2anime.utils.loss import _ImgTuple, DiscLoss, CycleGANLossGen
from selfie2anime.utils import visualize_loss_dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision.utils as vutils

import argparse
import torch
import torch.nn as nn
import wandb
import pickle

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
        ).to(DEVICE)
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
        wandb_run: wandb.Run,
        visualization_batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        batch_A, batch_B = visualization_batch
        img_grid_A = vutils.make_grid(batch_A, nrow=4, normalize=True)
        img_grid_B = vutils.make_grid(batch_B, nrow=4, normalize=True)

        wandb_run.log(
            {"Image Batch - A": wandb.Image(img_grid_A, caption="Real Images - A")}
        )
        wandb_run.log(
            {"Image Batch - B": wandb.Image(img_grid_B, caption="Real Images - B")}
        )

        batch_A = batch_A.to(DEVICE)
        batch_B = batch_B.to(DEVICE)

        self._load_model_state(
            path=f"{self.config.save_dir}/{self.config.model}/{self.config.mode}"
        )

        for epoch in range(1, self.config.num_epochs + 1):
            n = len(str(self.config.num_epochs))
            desc = f"Epoch [{epoch:0{n}d}/{self.config.num_epochs}]"
            num_batches = len(train_loader)
            iter = tqdm(train_loader, desc=desc, leave=False)

            disc_losses, gen_losses = self._train_epoch(iter, optim_disc, optim_gen)
            self._save_model_state(
                path=f"{self.config.save_dir}/{self.config.model}/{self.config.mode}"
            )

            with torch.no_grad():
                fake_batch_A = self.gen_A(batch_B).cpu()
                fake_batch_B = self.gen_B(batch_A).cpu()

            fake_img_grid_A = vutils.make_grid(fake_batch_A, nrow=4, normalize=True)
            fake_img_grid_B = vutils.make_grid(fake_batch_B, nrow=4, normalize=True)

            wandb_run.log(
                {
                    "Generated Image Batch - A": wandb.Image(
                        fake_img_grid_A, caption=f"Epoch {epoch} Generated Images - A"
                    )
                }
            )
            wandb_run.log(
                {
                    "Generated Image Batch - B": wandb.Image(
                        fake_img_grid_B, caption=f"Epoch {epoch} Generated Images - B"
                    )
                }
            )

            for key, val in disc_losses.items():
                if key not in self.disc_train_losses:
                    self.disc_train_losses[key] = []
                self.disc_train_losses[key].append(val / num_batches)

            for key, val in gen_losses.items():
                if key not in self.gen_train_losses:
                    self.gen_train_losses[key] = []
                self.gen_train_losses[key].append(val / num_batches)

        visualize_loss_dist(
            loss_history=self.disc_train_losses,
            title=f"{self.config.mode.capitalize()} CycleGAN Discriminator Training Loss-Epoch Distribution",
            save_path=f"{self.config.result_dir}/{self.config.model}/{self.config.mode}/disc_loss.png",
        )

        visualize_loss_dist(
            loss_history=self.gen_train_losses,
            title=f"{self.config.mode.capitalize()} CycleGAN Generator Training Loss-Epoch Distribution",
            save_path=f"{self.config.result_dir}/{self.config.model}/{self.config.mode}/gen_loss.png",
        )

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

            ## Train Discriminators
            optim_disc.zero_grad()
            disc_loss["loss_disc"].backward()
            optim_disc.step()

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

    def _save_model_state(self, path: str) -> None:
        """
        Save all four models and loss history as pickle files.
        """

        with open(f"{path}/gen_A.pkl", "wb") as file:
            pickle.dump(self.gen_A, file)
        with open(f"{path}/gen_B.pkl", "wb") as file:
            pickle.dump(self.gen_B, file)

        with open(f"{path}/disc_A.pkl", "wb") as file:
            pickle.dump(self.disc_A, file)
        with open(f"{path}/disc_B.pkl", "wb") as file:
            pickle.dump(self.disc_B, file)

        with open(f"{path}/gen_train_losses.pkl", "wb") as file:
            pickle.dump(self.gen_train_losses, file)
        with open(f"{path}/disc_train_losses.pkl", "wb") as file:
            pickle.dump(self.disc_train_losses, file)

    def _load_model_state(self, path: str) -> None:
        """
        Loads stored model state if any exists.
        """

        if not all(
            [
                os.path.exists(p)
                for p in [
                    f"{path}/gen_A.pkl",
                    f"{path}/gen_B.pkl",
                    f"{path}/disc_A.pkl",
                    f"{path}/disc_B.pkl",
                    f"{path}/gen_train_losses.pkl",
                    f"{path}/disc_train_losses",
                ]
            ]
        ):
            print("No saved checkpoint found, Training model from scratch ...")
            return

        print("Found Saved Checkpoint! Loading and continuing training ...")

        with open(f"{path}/gen_A.pkl", "wb") as file:
            self.gen_A = pickle.load(file)
        with open(f"{path}/gen_B.pkl", "wb") as file:
            self.gen_B = pickle.load(file)

        with open(f"{path}/disc_A.pkl", "wb") as file:
            self.disc_A = pickle.load(file)
        with open(f"{path}/disc_B.pkl", "wb") as file:
            self.disc_B = pickle.load(file)

        with open(f"{path}/gen_train_losses.pkl", "wb") as file:
            self.gen_train_losses = pickle.load(file)
        with open(f"{path}/disc_train_losses.pkl", "wb") as file:
            self.disc_train_losses = pickle.load(file)
