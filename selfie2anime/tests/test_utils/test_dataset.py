import unittest
import numpy as np
import torch

from selfie2anime.utils.dataset import Selfie2AnimeDataset
from torchvision import transforms

train_dataset = Selfie2AnimeDataset(root="data", split="train")
test_dataset = Selfie2AnimeDataset(root="data", split="test")


class TestDataset(unittest.TestCase):
    def test_len(self):
        # Train length = 3400
        # Test Length = 100

        assert len(train_dataset) == 3400
        assert len(test_dataset) == 100

    def test_image_size(self):
        # Image size = 256x256

        selfie, anime = train_dataset[0]
        assert np.array(selfie).shape == (256, 256, 3)
        assert np.array(anime).shape == (256, 256, 3)

        selfie, anime = test_dataset[0]
        assert np.array(selfie).shape == (256, 256, 3)
        assert np.array(anime).shape == (256, 256, 3)

    def test_transform(self):
        train_dataset.transform = transforms.ToTensor()
        test_dataset.transform = transforms.ToTensor()

        selfie, anime = train_dataset[0]
        assert isinstance(selfie, torch.Tensor) and selfie.size() == (3, 256, 256)
        assert isinstance(anime, torch.Tensor) and anime.size() == (3, 256, 256)

        selfie, anime = test_dataset[0]
        assert isinstance(selfie, torch.Tensor) and selfie.size() == (3, 256, 256)
        assert isinstance(anime, torch.Tensor) and anime.size() == (3, 256, 256)
