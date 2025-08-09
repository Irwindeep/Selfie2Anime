from selfie2anime import CycleGAN
from selfie2anime.utils.dataset import Selfie2AnimeDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import argparse
import torch


def train_cyclegan(config: argparse.Namespace) -> None:
    transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = Selfie2AnimeDataset(
        root=config.dataset_root, split="train", transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    cyclegan = CycleGAN(config=config, mode=config.mode)

    optim_disc = torch.optim.Adam(
        params=list(cyclegan.disc_A.parameters()) + list(cyclegan.disc_B.parameters()),
        lr=config.disc_lr,
        betas=(config.beta1, config.beta2),
    )
    optim_gen = torch.optim.Adam(
        params=list(cyclegan.gen_A.parameters()) + list(cyclegan.gen_B.parameters()),
        lr=config.gen_lr,
        betas=(config.beta1, config.beta2),
    )

    cyclegan.train(train_loader, optim_disc=optim_disc, optim_gen=optim_gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cyclegan")

    if parser.parse_args().model == "cyclegan":
        parser = argparse.ArgumentParser()

        parser.add_argument("--mode", type=str, default="vanilla")
        parser.add_argument("--dataset_root", type=str, default="data")
        parser.add_argument("--img_size", type=int, default=256)
        parser.add_argument("--batch_size", type=int, default=4)

        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--num_features", type=int, default=64)
        parser.add_argument("--num_residuals", type=int, default=9)
        parser.add_argument("--disc_channels", type=list, default=[64, 128, 256, 512])

        parser.add_argument("--init_mean", type=float, default=0.0)
        parser.add_argument("--init_std", type=float, default=0.02)

        parser.add_argument("--num_epochs", type=int, default=100)
        parser.add_argument("--disc_lr", type=float, default=2e-4)
        parser.add_argument("--gen_lr", type=float, default=2e-4)
        parser.add_argument("--beta1", type=float, default=0.5)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--lmbd_cyc", type=float, default=10.0)
        parser.add_argument("--lmbd_iden", type=float, default=0.1)

        config = parser.parse_args()
        train_cyclegan(config)
