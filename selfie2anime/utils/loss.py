from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


_ImgTuple = Tuple[torch.Tensor, torch.Tensor]


class DiscLoss(nn.Module):
    """
    Implementation of discriminator Loss for CycleGAN - all modes.
    """

    def __init__(self) -> None:
        super(DiscLoss, self).__init__()
        self.mse = nn.MSELoss()

    # naming convention: A is refered for selfie and B for anime
    def forward(
        self, disc_real: _ImgTuple, disc_fake: _ImgTuple
    ) -> Dict[str, torch.Tensor]:
        disc_real_A, disc_real_B = disc_real
        disc_fake_A, disc_fake_B = disc_fake

        ## Discriminator Loss is divided by 2 to stabilize training
        loss_disc_real_A = self.mse(disc_real_A, torch.ones_like(disc_real_A))
        loss_disc_fake_A = self.mse(disc_fake_A, torch.zeros_like(disc_fake_A))
        loss_disc_A = (loss_disc_real_A + loss_disc_fake_A) / 2

        loss_disc_real_B = self.mse(disc_real_B, torch.ones_like(disc_real_B))
        loss_disc_fake_B = self.mse(disc_fake_B, torch.zeros_like(disc_fake_B))
        loss_disc_B = (loss_disc_real_B + loss_disc_fake_B) / 2

        output = {
            "loss_disc": loss_disc_A + loss_disc_B,
            "loss_disc_A": loss_disc_A.detach(),
            "loss_disc_B": loss_disc_B.detach(),
        }
        return output


class CycleGANLossGen(nn.Module):
    """
    Loss function for Vanilla CycleGAN Generator as described in -
    "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
    This implementation covers three types of losses described originally i.e., `loss_gan`, `loss_cyc` and `loss_iden`
    for GAN Loss, Cycle Consistency Loss and Identity Loss.

    NOTE: Identity loss is a computationally expensive option, the primary goal of a CycleGAN is to address Cycle Consistency loss,
    hence Identity loss computation can be avoided by setting `iden_imgs = None`.
    """

    def __init__(self, lmbd_cyc: float = 10.0, lmbd_iden: float = 0.0) -> None:
        super(CycleGANLossGen, self).__init__()

        self.lmbd_cyc = lmbd_cyc
        self.lmbd_iden = lmbd_iden

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    # naming convention: A is refered for selfie and B for anime
    def forward(
        self,
        real_imgs: _ImgTuple,
        cyc_imgs: _ImgTuple,
        disc_fake: _ImgTuple,
        iden_imgs: Optional[_ImgTuple] = None,
    ) -> Dict[str, torch.Tensor]:
        real_A, real_B = real_imgs

        cyc_A, cyc_B = cyc_imgs
        disc_fake_A, disc_fake_B = disc_fake

        ## GAN Loss - using discriminator
        loss_gen_A = self.mse(disc_fake_A, torch.ones_like(disc_fake_A))
        loss_gen_B = self.mse(disc_fake_B, torch.ones_like(disc_fake_B))

        ## Cycle consistency Loss
        loss_cyc_A = self.l1(cyc_A, real_A)
        loss_cyc_B = self.l1(cyc_B, real_B)

        ## Identity Loss (optional)
        loss_iden_A = torch.tensor(0.0, device=real_A.device)
        loss_iden_B = torch.tensor(0.0, device=real_B.device)
        if iden_imgs:
            iden_A, iden_B = iden_imgs
            loss_iden_A = self.l1(iden_A, real_A)
            loss_iden_B = self.l1(iden_B, real_B)

        loss_gen = (
            (loss_gen_A + loss_gen_B)
            + (loss_cyc_A + loss_cyc_B) * self.lmbd_cyc
            + (loss_iden_A + loss_iden_B) * self.lmbd_iden
        )

        output = {
            "loss_gen": loss_gen,
            "loss_gan": (loss_gen_A + loss_gen_B).detach(),
            "loss_cyc": (loss_cyc_A + loss_cyc_B).detach(),
            "loss_iden": (loss_iden_A + loss_iden_B).detach(),
        }
        return output
