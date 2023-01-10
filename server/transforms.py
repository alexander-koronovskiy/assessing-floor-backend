import torch
from torch import nn, Tensor
from torchvision.transforms import (
    functional as F,
    transforms as T
)

from .exceptions import TransformError


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class ToTensor(nn.Module):
    def forward(self, image):
        return F.to_tensor(image)


class Resize(nn.Module):
    def __init__(self, size: int = 1024):
        super().__init__()
        self.size = size
        self.resize = T.Resize((size, size))

    def forward(self, image: Tensor):
        if isinstance(image, torch.Tensor):
            if image.ndim not in {2, 3}:
                raise TransformError('Image should be 2/3 dimensional')
            elif image.ndim == 2:
                image = image.unsqueeze(0)
        return self.resize(image)


class Pad(nn.Module):
    def __init__(self, fill: int = 0, padding_mode: str = 'constant'):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def _get_padding_sequence(image: Tensor):
        width, height = F.get_image_size(image)

        flag = height - width
        if flag == 0:
            return 0, 0, 0, 0

        diff = abs(flag)
        border_1, border_2 = diff // 2, round(diff / 2)

        if flag > 0:
            top, bottom = 0, 0
            left, right = border_1, border_2
        else:
            top, bottom = border_1, border_2
            left, right = 0, 0
        return left, top, right, bottom

    def forward(self, image: Tensor):
        if isinstance(image, torch.Tensor):
            if image.ndim not in {2, 3}:
                raise TransformError('Image should be 2/3 dimensional')
            elif image.ndim == 2:
                image = image.unsqueeze(0)

        padding = self._get_padding_sequence(image)
        pad = T.Pad(padding, fill=self.fill, padding_mode=self.padding_mode)
        return pad(image)
