import os
import torch
from PIL import Image


class Pix2PixDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, input_first=True):
        super(Pix2PixDataset, self).__init__()
        self.root = root
        self.files = sorted(os.listdir(root))
        self.transforms = transforms
        self.input_first = input_first

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.files[index]))
        tensor = self.transforms(image)
        width = tensor.shape[-1] // 2

        if self.input_first:
            return tensor[..., :width], tensor[..., width:]

        return tensor[..., width:], tensor[..., :width]
