from typing import Dict

from selfie2anime.utils import dataset
from selfie2anime.utils import conv
from selfie2anime.utils import loss

import matplotlib.pyplot as plt


def visualize_loss_dist(
    loss_history: Dict[str, list], title: str, save_path: str
) -> None:
    """
    Visualize Loss Distributions of each model.
    """

    for key, history in loss_history.items():
        plt.plot(range(1, len(history) + 1), history, label=key)

    plt.title(title, fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(save_path)


__all__ = [
    "conv",
    "dataset",
    "loss",
    "visualize_loss_dist",
]
