import torch
from ..models.loss import AdversarialLoss


class Trainer(object):
    def __init__(self, params, experimenter, generator, discriminator=None):
        self.params = params
        self.experimenter = experimenter
        self.generator = generator
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=params.lr)

        self.discriminator = discriminator
        if params.adversarial:
            self.discr_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=params.lr)

        self.l1_criterion = torch.nn.L1Loss()
        if params.adversarial:
            self.criterion = AdversarialLoss(params.loss, params.loss_lambda)
        elif params.loss == 'L1':
            self.criterion = torch.nn.L1Loss()
        elif params.loss == 'L2':
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError('Unknown loss type')

    def process_epoch(self, loader, train=True):
        self.generator.train() if train else self.generator.eval()

        running_loss, running_l1 = 0.0, 0.0

        for inputs, targets in loader:
            inputs = inputs.to(self.params.device)
            targets = targets.to(self.params.device)

            with torch.set_grad_enabled(train):
                self.gen_optimizer.zero_grad()
                outputs = self.generator(inputs)
                loss = self.criterion(outputs, targets)
                l1 = self.l1_criterion(outputs, targets)

            if train:
                loss.backward()
                self.gen_optimizer.step()

            running_loss += loss.item() * inputs.shape[0]
            running_l1 += l1.item() * inputs.shape[0]

        running_loss /= len(loader.dataset)
        running_l1 /= len(loader.dataset)

        return running_loss, running_l1

    def process_epoch_adversarial(self, loader, train=True):
        self.generator.train() if train else self.generator.eval()
        self.discriminator.train() if train else self.discriminator.eval()

        running_discr_loss, running_gen_loss, running_l1 = 0.0, 0.0, 0.0

        for inputs, targets in loader:
            inputs = inputs.to(self.params.device)
            targets = targets.to(self.params.device)

            with torch.set_grad_enabled(train):
                # discriminator training
                self.discr_optimizer.zero_grad()
                outputs = self.generator(inputs).detach()
                logits = self.discriminator(torch.cat([inputs, outputs], dim=1))
                fake_loss = self.criterion.discriminator_loss(logits, real=False)

                logits = self.discriminator(torch.cat([inputs, targets], dim=1))
                real_loss = self.criterion.discriminator_loss(logits, real=True)
                discr_loss = 0.5 * (fake_loss + real_loss)

                if train:
                    discr_loss.backward()
                    self.discr_optimizer.step()

                # generator training
                self.gen_optimizer.zero_grad()
                outputs = self.generator(inputs)
                logits = self.discriminator(torch.cat([inputs, outputs], dim=1))
                gen_loss = self.criterion.generator_loss(outputs, targets, logits)
                l1_loss = self.l1_criterion(outputs, targets)

                if train:
                    gen_loss.backward()
                    self.gen_optimizer.step()

                running_discr_loss += discr_loss.item() * inputs.shape[0]
                running_gen_loss += gen_loss.item() * inputs.shape[0]
                running_l1 += l1_loss.item() * inputs.shape[0]

        running_discr_loss /= len(loader.dataset)
        running_gen_loss /= len(loader.dataset)
        running_l1 /= len(loader.dataset)

        return running_discr_loss, running_gen_loss, running_l1

    def train(self, train_loader, valid_loader):
        for epoch in range(1, self.params.num_epochs + 1):
            if not self.params.adversarial:
                train_metrics = self.process_epoch(train_loader, train=True)
                valid_metrics = self.process_epoch(valid_loader, train=False)
            else:
                train_metrics = self.process_epoch_adversarial(train_loader, train=True)
                valid_metrics = self.process_epoch_adversarial(valid_loader, train=False)

            self.experimenter.generate_examples(epoch, train_metrics, valid_metrics)
            if self.params.save_checkpoints and epoch % self.params.checkpoints_freq == 0:
                self.experimenter.save_checkpoint(epoch)

