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
    train_dataset = Pix2PixDataset(root=os.path.join(data_root, params.train_suffix),
                                   size=params.image_size, input_left=params.input_left,
                                   flip=params.flip, normalize=params.normalize)
    valid_dataset = Pix2PixDataset(root=os.path.join(data_root, params.valid_suffix),
                                   size=params.image_size, input_left=params.input_left,
                                   normalize=params.normalize)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size,
                                               num_workers=params.num_workers, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=params.batch_size,
                                               num_workers=params.num_workers, shuffle=False, pin_memory=True)
    if params.verbose:
        print('Dataloaders prepared')

    # build models
    generator = build_generator(params).to(params.device)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=params.lr)
    discriminator, discr_optimizer = None, None
    if params.adversarial:
        discriminator = build_discriminator(params).to(params.device)
        discr_optimizer = torch.optim.Adam(discriminator.parameters(), lr=params.lr)

    if params.verbose:
        print('Models initialized')

    # train models
    experimenter = Experimenter(params, valid_loader, generator, discriminator,
                                gen_optimizer, discr_optimizer)
    trainer = Trainer(params, experimenter, generator, discriminator,
                      gen_optimizer, discr_optimizer)
    trainer.train(train_loader, valid_loader)


if __name__ == '__main__':
    main()
