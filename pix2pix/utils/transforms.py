import torch
import torchvision


class Pix2PixTransforms(object):
    def __init__(self, size, flip=False, normalize=False, scale=None):
        self.transforms = [torchvision.transforms.Resize(size)]

        if flip:
            self.transforms += [torchvision.transforms.RandomHorizontalFlip(0.5)]
        if scale:
            self.transforms += [torchvision.transforms.RandomResizedCrop(size, scale, ratio=(1.0, 1.0))]
        if normalize:
            self.transforms += [torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transforms = torchvision.transforms.Compose(self.transforms)

    def __call__(self, inputs, outputs):
        images = torch.stack([inputs, outputs], dim=0)
        images = self.transforms(images)
        return images[0], images[1]
