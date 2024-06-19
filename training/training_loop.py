import torch
import csv
from tqdm.auto import tqdm
import os
import numpy as np


class TrainingLoop:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch=0,
                 save_path: str = ''
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.save_path = save_path

        self.training_loss = []
        self.validation_loss = [np.nan, np.nan]
        self.learning_rate = []
        self.csv_path = os.path.join(save_path, 'history.csv')
        self.model_path = os.path.join(save_path, 'best.pt')
        self.fieldnames = ['training_losses', 'validation_losses', 'lr_rates']

        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(self.fieldnames)

    def run_training(self):

        #progressbar = tqdm(range(self.epochs), desc='Progress')

        for i in tqdm(range(self.epochs)):
            self.epoch += 1
            self._train()

            if self.validation_DataLoader is not None:
                self._validate()

            # with open(self.csv_path, 'a') as f:
            #     writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            #     writer.writerow({'training_losses': str(self.training_loss[-1]),
            #                      'validation_losses': str(self.validation_loss[-1]),
            #                      'lr_rates': str(self.learning_rate[-1])})
            # if i > 2:
            #     if np.mean(self.validation_loss[-1]) <= np.min(self.validation_loss[:-1]):
            #         torch.save(self.model.state_dict(), self.model_path)
            # torch.save(self.model.state_dict(),
            #            self.model_path.replace('best', 'last'))

            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    # learning rate scheduler step with validation loss
                    self.lr_scheduler.batch(self.validation_loss[i])
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        torch.save(self.model.state_dict(),
                   self.model_path.replace('best', 'last'))
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        self.model.train()
        train_losses = []

        for (x, y) in self.training_DataLoader:
            try:
                input, target = x.to(self.device), y.to(
                    self.device)  # send to device (GPU or CPU)
                self.optimizer.zero_grad()  # zerograd the parameters
                out = self.model(input)  # one forward pass
                loss = self.criterion(out, target)  # calculate loss
                loss_value = loss.item()
                train_losses.append(loss_value)
                loss.backward()  # one backward pass
                self.optimizer.step()  # update the parameters
            except:
                print('error')

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

    def _validate(self):

        self.model.eval()
        valid_losses = []

        for (x, y) in self.validation_DataLoader:

            input, target = x.to(self.device), y.to(
                self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

        self.validation_loss.append(np.mean(valid_losses))
