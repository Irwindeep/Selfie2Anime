import os

from typing import Any, Callable, Literal, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image


class Selfie2AnimeDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "test"] = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Loads the Selfie2Anime kaggle dataset `https://www.kaggle.com/datasets/arnaud58/selfie2anime` as a pytorch dataset.
        """
        self.selfie_root = os.path.join(root, f"{split}A")
        self.anime_root = os.path.join(root, f"{split}B")

        self.selfie_paths = os.listdir(self.selfie_root)
        self.anime_paths = os.listdir(self.anime_root)

        self.transform = transform

    def __len__(self) -> int:
        return max(len(self.selfie_paths), len(self.anime_paths))

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        selfie_path = self.selfie_paths[idx % len(self.selfie_paths)]
        anime_path = self.anime_paths[idx % len(self.anime_paths)]

        selfie = Image.open(os.path.join(self.selfie_root, selfie_path))
        anime = Image.open(os.path.join(self.anime_root, anime_path))

        if self.transform:
            selfie = self.transform(selfie)
            anime = self.transform(anime)

        return selfie, anime
