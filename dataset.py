from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from utils import CircleParams, CircleParamsScaler, generate_examples


def generate_circle(
    img_size: int = 100,
    min_radius: int = 10,
    max_radius: int = 50,
    min_noise: float = 0,
    max_noise: float = 1,
) -> CircleParams:
    """
    Returns a single circle with a noise level sampled from a uniform distribution
    """
    noise = np.random.uniform(low=min_noise, high=max_noise)
    gen = generate_examples(
        noise_level=noise,
        img_size=img_size,
        min_radius=min_radius,
        max_radius=max_radius,
    )
    return next(gen)


class CircleImageDataset(Dataset):
    def __init__(
        self,
        img_size: int = 100,
        min_radius: int = 10,
        max_radius: int = 50,
        min_noise: float = 0,
        max_noise: float = 1,
        dataset_size: int = 1000,
        img_transform: Optional[StandardScaler] = None,
        fixed: bool = False,
        device: str = "cpu",
    ) -> None:
        self.img_size = img_size
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.dataset_size = dataset_size
        self.fixed = fixed
        self.device = device

        # Take a sample to determine STD of noise
        if img_transform is None:
            samples = []
            for _ in range(1000):
                noise = np.random.uniform(low=self.min_noise, high=self.max_noise)
                gen = generate_examples(noise_level=noise)
                img, _ = next(gen)
                samples.append(img)
            sample_std = float(np.std(np.array(samples)))

            img_transform = StandardScaler()
            img_transform.scale_ = sample_std
            img_transform.mean_ = 0.5  # From noisy_circle

        self.img_transform = img_transform

        self.circle_transform = CircleParamsScaler(
            img_size=img_size, min_radius=min_radius, max_radius=max_radius
        )

        self.data = []
        if self.fixed:
            for _ in range(self.dataset_size):
                self.data.append(
                    generate_circle(
                        img_size=self.img_size,
                        min_radius=self.min_radius,
                        max_radius=self.max_radius,
                        min_noise=self.min_noise,
                        max_noise=self.max_noise,
                    )
                )

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index) -> Tuple[torch.Tensor, CircleParams]:
        if self.fixed:
            img, circle = self.data[index]
        else:
            img, circle = generate_circle(
                img_size=self.img_size,
                min_radius=self.min_radius,
                max_radius=self.max_radius,
                min_noise=self.min_noise,
                max_noise=self.max_noise,
            )

        img = self.img_transform.transform(img.reshape(1, self.img_size**2)).reshape(
            1, self.img_size, self.img_size
        )
        circle = self.circle_transform.transform(circle)
        return torch.FloatTensor(img).to(self.device), torch.FloatTensor(
            [circle.radius, circle.col, circle.row]
        ).to(self.device)
