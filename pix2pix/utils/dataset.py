import os
import random
import torch
from torchvision import transforms
from PIL import Image


class Pix2PixDataset(torch.utils.data.Dataset):
    def __init__(self, root, size, input_left=True, flip=False, normalize=False):
        super(Pix2PixDataset, self).__init__()
        self.root = root
        self.files = sorted(os.listdir(root))
        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
        self.input_left = input_left
        self.flip = flip
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.files[index]))
        tensor = self.to_tensor(self.resize(image))
        width = tensor.shape[-1] // 2

        if self.input_left:
            inputs, outputs = tensor[..., :width], tensor[..., width:]
        else:
            inputs, outputs = tensor[..., width:], tensor[..., :width]

        if self.flip and random.random() > 0.5:
            inputs, outputs = torch.flip(inputs, dims=[-1]), torch.flip(outputs, dims=[-1])

        if self.normalize:
            inputs, outputs = (inputs - 0.5) / 0.5, (outputs - 0.5) / 0.5

        return inputs, outputs
