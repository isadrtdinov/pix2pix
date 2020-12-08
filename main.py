import os
import torch
import torchvision
from config import set_params
from pix2pix.utils import (
    Experimenter,
    Pix2PixDataset,
    Pix2PixTransforms,
    set_random_seed
)
from pix2pix.train import Trainer
from pix2pix.models import build_generator, build_discriminator


def main():
    # set params and random seed
    params = set_params()
    set_random_seed(params.random_seed)

    params.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if params.verbose:
        print('Using device', params.device)

    # load dataset
    data_root = os.path.join(params.datasets_dir, params.dataset)
    if not os.path.isdir(data_root):
        os.system('./scripts/download_data.sh ' + params.dataset)

    # init datasets and dataloader
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(params.image_size),
        torchvision.transforms.ToTensor()
    ])

    train_transforms = Pix2PixTransforms(params.image_size, flip=params.flip,
                                         normalize=params.normalize)
    valid_transforms = Pix2PixTransforms(params.image_size, normalize=params.normalize)

    train_dataset = Pix2PixDataset(root=os.path.join(data_root, params.train_suffix),
                                   input_left=params.input_left, transforms=train_transforms)
    valid_dataset = Pix2PixDataset(root=os.path.join(data_root, params.valid_suffix),
                                   input_left=params.input_left, transforms=valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size,
                                               num_workers=params.num_workers, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=params.batch_size,
                                               num_workers=params.num_workers, shuffle=False, pin_memory=True)
    if params.verbose:
        print('Dataloaders prepared')

    # build models
    generator = build_generator(params).to(params.device)
    discriminator = None
    if params.adversarial:
        discriminator = build_discriminator(params)

    if params.verbose:
        print('Models initialized')

    # train models
    experimenter = Experimenter(params, valid_loader, generator, discriminator)
    trainer = Trainer(params, experimenter, generator, discriminator)
    trainer.train(train_loader, valid_loader)


if __name__ == '__main__':
    main()
