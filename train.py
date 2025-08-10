from selfie2anime import CycleGAN
from selfie2anime.utils.dataset import Selfie2AnimeDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dotenv import load_dotenv

import argparse
import torch
import os
import wandb

torch.manual_seed(12)


def wandb_setup(config: argparse.Namespace) -> wandb.Run:
    load_dotenv(config.env_path)
    wandb_key = os.getenv(config.wandb_key)
    wandb.login(key=wandb_key)

    wandb_run = wandb.init(project="selfie2anime-cyclegan-training")
    return wandb_run


def train_cyclegan(config: argparse.Namespace, wandb_run: wandb.Run) -> None:
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

    indices = torch.randint(low=0, high=len(train_dataset) - 1, size=(4 * 4,)).numpy()
    selfie_batch = torch.stack([train_dataset[idx][0] for idx in indices])
    anime_batch = torch.stack([train_dataset[idx][1] for idx in indices])

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
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(f"{config.save_dir}/{config.model}", exist_ok=True)
    os.makedirs(f"{config.save_dir}/{config.model}/{config.mode}", exist_ok=True)

    os.makedirs(config.result_dir, exist_ok=True)
    os.makedirs(f"{config.result_dir}/{config.model}", exist_ok=True)
    os.makedirs(f"{config.result_dir}/{config.model}/{config.mode}", exist_ok=True)

    cyclegan.train(
        train_loader=train_loader,
        optim_disc=optim_disc,
        optim_gen=optim_gen,
        wandb_run=wandb_run,
        visualization_batch=(selfie_batch, anime_batch),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cyclegan")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--result_dir", type=str, default="results")

    # make sure env has a .env file with wandb API key
    parser.add_argument("--env_path", type=str, default=".env")
    parser.add_argument("--wandb_key", type=str, default="WANDB_KEY")

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
    wandb_run = wandb_setup(config)

    if config.model == "cyclegan":
        train_cyclegan(config, wandb_run)
