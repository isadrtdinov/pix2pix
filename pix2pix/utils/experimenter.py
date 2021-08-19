import os
import json
import time
import torch
from torchvision.transforms import ToPILImage
from .utils import count_params


class Experimenter(object):
    def __init__(self, params, valid_loader, generator, discriminator,
                 gen_optimizer, discr_optimizer):
        self.params = params
        self.start_time = time.time()
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.discr_optimizer = discr_optimizer

        self.to_image = ToPILImage()
        self.metrics = {'train L1': [], 'valid L1': []}
        if self.params.adversarial:
            self.metrics.update({'train discr loss': [], 'train gen loss': [],
                                 'valid discr loss': [], 'valid gen loss': []})
        else:
            self.metrics.update({'train loss': [], 'valid loss': []})

        if not os.path.isdir(params.runs_dir):
            os.mkdir(params.runs_dir)

        run_id = max((int(run) for run in os.listdir(params.runs_dir)), default=0) + 1
        self.run_path = os.path.join(params.runs_dir, '{:02d}'.format(run_id))
        os.mkdir(self.run_path)
        if params.verbose:
            print('Logging run at {}'.format(self.run_path))

        if params.save_checkpoints:
            self.checkpoints_subdir = os.path.join(self.run_path, params.checkpoints_subdir)
            os.mkdir(self.checkpoints_subdir)

        if params.load_checkpoint is not None:
            state_dict = torch.load(params.load_checkpoint)
            self.generator.load_state_dict(state_dict['generator'])
            self.gen_optimizer.load_state_dict(state_dict['gen_optimizer'])
            if params.adversarial:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                self.discr_optimizer.load_state_dict(state_dict['discr_optimizer'])

        examples_subdir = os.path.join(self.run_path, params.examples_subdir)
        os.mkdir(examples_subdir)
        self.examples_subdirs = []
        for example_id in params.examples_ids:
            self.examples_subdirs += [os.path.join(examples_subdir, str(example_id))]
            os.mkdir(self.examples_subdirs[-1])

        self.write_metadata(params, generator, discriminator)

        self.example_inputs = []
        for subdir, example_id in zip(self.examples_subdirs, params.examples_ids):
            inputs, outputs = valid_loader.dataset[example_id]
            self.example_inputs += [inputs]

            inputs_file = str(example_id) + '_input.jpg'
            self.to_image(inputs * 0.5 + 0.5).save(os.path.join(subdir, inputs_file), 'JPEG')

            outputs_file = str(example_id) + '_output.jpg'
            self.to_image(outputs * 0.5 + 0.5).save(os.path.join(subdir, outputs_file), 'JPEG')

        self.example_inputs = torch.stack(self.example_inputs)

    def write_metadata(self, params, generator, discriminator):
        metadata = {
            'dataset': params.dataset,
            'num epochs': params.num_epochs,
            'batch size': params.batch_size,
            'learning rate': params.lr,
            'loss': params.loss,
            'generator channels': params.generator_channels,
            'generator layers:': params.generator_layers,
            'generator kernel': params.generator_kernel,
            'generator dropout': params.generator_dropout,
            'generator norm': params.generator_norm,
            'generator params': count_params(generator)
        }

        if params.adversarial:
            metadata.update({
                'discriminator channels': params.discriminator_channels,
                'discriminator layers': params.discriminator_layers,
                'discriminator norm': params.discriminator_norm,
                'discriminator params': count_params(discriminator),
                'loss lambda': params.loss_lambda
            })

        metadata_path = os.path.join(self.run_path, params.metadata_file)
        with open(metadata_path, 'w') as metadata_file:
            json.dump(metadata, metadata_file)

    def generate_examples(self, epoch, train_metrics, valid_metrics):
        with torch.no_grad():
            example_outputs = self.generator(self.example_inputs.to(self.params.device)).cpu()
            if self.params.normalize:
                example_outputs = example_outputs * 0.5 + 0.5

        for subdir, example_id, example in zip(self.examples_subdirs, self.params.examples_ids, example_outputs):
            example_file = str(example_id) + '_' + str(epoch) + '.jpg'
            self.to_image(example).save(os.path.join(subdir, example_file))

        if self.params.adversarial:
            self.metrics['train discr loss'] += [train_metrics[0]]
            self.metrics['train gen loss'] += [train_metrics[1]]
            self.metrics['valid discr loss'] += [valid_metrics[0]]
            self.metrics['valid gen loss'] += [valid_metrics[1]]
        else:
            self.metrics['train loss'] += [train_metrics[0]]
            self.metrics['valid loss'] += [valid_metrics[1]]

        self.metrics['train L1'] += [train_metrics[-1]]
        self.metrics['valid L1'] += [valid_metrics[-1]]

        train_metrics = '(' + ', '.join('{:.4f}'.format(value) for value in train_metrics) + ')'
        valid_metrics = '(' + ', '.join('{:.4f}'.format(value) for value in valid_metrics) + ')'

        if self.params.verbose:
            print('{}/{} {}s\n\ttrain metrics = {}\n\tvalid metrics = {}'.format(
                epoch, self.params.num_epochs, int(time.time() - self.start_time),
                train_metrics, valid_metrics
            ))

    def save_checkpoint(self, epoch):
        state_dict = {
            'generator': self.generator.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict()
        }
        if self.params.adversarial:
            state_dict.update({
                'discriminator': self.discriminator.state_dict(),
                'discr_optimizer': self.discr_optimizer.state_dict()
            })

        checkpoint_file = self.params.checkpoints_template.format(epoch)
        checkpoint_file = os.path.join(self.checkpoints_subdir, checkpoint_file)
        torch.save(state_dict, checkpoint_file)

    def __del__(self):
        metrics_path = os.path.join(self.run_path, self.params.metrics_file)
        with open(metrics_path, 'w') as metrics_file:
            json.dump(self.metrics, metrics_file)
