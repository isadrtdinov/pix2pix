import torch


class Trainer(object):
    def __init__(self, params, experimenter, generator, discriminator=None):
        self.params = params
        self.experimenter = experimenter
        self.generator = generator
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=params.lr)

        self.discriminator = discriminator
        if params.adversarial:
            pass

        self.l1_criterion = torch.nn.L1Loss()
        if params.adversarial:
            self.criterion = None
        elif params.loss == 'L1':
            self.criterion = torch.nn.L1Loss()
        elif params.loss == 'L2':
            self.criterion = torch.nn.MSELoss()

    def process_epoch(self, loader, train=True):
        running_loss, running_l1 = 0.0, 0.0

        for inputs, targets in loader:
            with torch.set_grad_enabled(train):
                inputs = inputs.to(self.params.device)
                targets = targets.to(self.params.device)

                outputs = self.generator(inputs)
                loss = self.criterion(outputs, targets)
                l1 = self.l1_criterion(outputs, targets)

            if train:
                loss.backward()
                self.gen_optimizer.step()

            running_loss += loss * inputs.shape[0]
            running_l1 += l1 * inputs.shape[0]

        running_loss /= len(loader.dataset)
        running_l1 /= len(loader.dataset)

        return running_loss, running_l1

    def process_epoch_adversarial(self, loader, train=True):
        return 0.0, 0.0

    def train(self, train_loader, valid_loader):
        for epoch in range(1, self.params.num_epochs + 1):
            if not self.params.adversarial:
                train_metrics = self.process_epoch(train_loader, train=True)
                valid_metrics = self.process_epoch(valid_loader, train=False)
            else:
                train_metrics = self.process_epoch_adversarial(train_loader, train=True)
                valid_metrics = self.process_epoch_adversarial(valid_loader, train=False)

            self.experimenter.generate_examples(epoch, train_metrics, valid_metrics)
            self.experimenter.save_checkpoint(epoch)

            if self.params.verbose:
                print('{}/{} train loss: {}, valid loss: {}'.format(
                    epoch, self.params.num_epochs, train_metrics[0], valid_metrics[0]
                ))
