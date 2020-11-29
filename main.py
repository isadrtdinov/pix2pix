import os
import torch
import torchvision
from config import set_params
from pix2pix.utils import (
    Experimenter,
    Pix2PixDataset,
    set_random_seed
)
from pix2pix.train import Trainer
from pix2pix.models import build_generator


def main():
    # set params and random seed
    params = set_params()
    set_random_seed(params.random_seed)

    params.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if params.verbose:
        print('Using device', params.device)

    # load dataset
    os.system('./scripts/download_data.sh ' + params.dataset)
    data_root = os.path.join(params.datasets_dir, params.dataset)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(params.image_size),
        torchvision.transforms.ToTensor()
    ])

    # init datasets and dataloader
    train_dataset = Pix2PixDataset(os.path.join(data_root, params.train_suffix),
                                   transforms, params.input_first)
    valid_dataset = Pix2PixDataset(os.path.join(data_root, params.valid_suffix),
                                   transforms, params.input_first)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size,
                                               num_workers=params.num_workers, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=params.batch_size,
                                               num_workers=params.num_workers, shuffle=False, pin_memory=True)
    if params.verbose:
        print('Dataloader prepared')

    generator = build_generator(params)
    discriminator = None
    if params.adversarial:
        pass
    if params.verbose:
        print('Model initialized')

    experimenter = Experimenter(params, valid_loader, generator, discriminator)
    trainer = Trainer(params, experimenter, generator, discriminator)
    trainer.train(train_loader, valid_loader)


if __name__ == '__main__':
    main()
