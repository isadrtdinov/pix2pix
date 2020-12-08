import torch
from torch import nn


class AdversarialLoss(nn.Module):
    def __init__(self, reconstruction_loss='L1', loss_lambda=1.0,
                 real_label=1.0, fake_label=0.0):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        if reconstruction_loss == 'L1':
            self.loss = nn.L1Loss()
        elif reconstruction_loss == 'L2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError('Unknown loss type')

        self.loss_lambda = loss_lambda
        self.real_label = real_label
        self.fake_label = fake_label

    def forward(self, outputs, targets, logits):
        return self.generator_loss(outputs, targets, logits)

    def generator_loss(self, outputs, targets, logits):
        labels = torch.tensor([self.real_label]).expand_as(logits).to(logits.device)
        adversarial_loss = self.bce(logits, labels)
        reconstruction_loss = self.loss(outputs, targets)
        return adversarial_loss + self.loss_lambda * reconstruction_loss

    def discriminator_loss(self, logits, real=True):
        if real:
            labels = torch.tensor([self.real_label]).expand_as(logits).to(logits.device)
        else:
            labels = torch.tensor([self.fake_label]).expand_as(logits).to(logits.device)

        return self.bce(logits, labels)
