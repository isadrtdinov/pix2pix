import os
import torch
import torchvision
from PIL import Image


class Pix2PixDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, input_left=True):
        super(Pix2PixDataset, self).__init__()
        self.root = root
        self.files = sorted(os.listdir(root))
        self.to_tensor = torchvision.transforms.ToTensor()
        self.transforms = transforms
        self.input_left = input_left

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.files[index]))
        tensor = self.to_tensor(image)
        width = tensor.shape[-1] // 2

        if self.input_left:
            inputs, outputs = tensor[..., :width], tensor[..., width:]
        else:
            inputs, outputs = tensor[..., width:], tensor[..., :width]

        if self.transforms:
            inputs, outputs = self.transforms(inputs, outputs)

        return inputs, outputs