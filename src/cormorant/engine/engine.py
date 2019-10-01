import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as sched

import argparse, os, sys, pickle
from datetime import datetime
from math import sqrt, inf, log, log2, exp, ceil

MAE = torch.nn.L1Loss()
MSE = torch.nn.MSELoss()
RMSE = lambda x, y : sqrt(MSE(x, y))

import logging
logger = logging.getLogger(__name__)

class Engine:
    """
    Class for both training and inference phasees of the Cormorant network.

    Includes checkpoints, optimizer, scheduler.

    Roughly based upon TorchNet
    """
    def __init__(self, args, dataloaders, model, loss_fn, optimizer, scheduler, restart_epochs, device, dtype):
        self.args = args
        self.dataloaders = dataloaders
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.restart_epochs = restart_epochs

        self.stats = dataloaders['train'].dataset.stats

        # TODO: Fix this until TB summarize is implemented.
        self.summarize = False

        self.best_loss = inf
        self.epoch = 0
        self.minibatch = 0

        self.device = device
        self.dtype = dtype

    def _save_checkpoint(self, valid_mae):
        if not self.args.save: return

        save_dict = {'args': self.args,
                     'model_state': self.model.state_dict(),
                     'optimizer_state': self.optimizer.state_dict(),
                     'scheduler_state': self.scheduler.state_dict(),
                     'epoch': self.epoch,
                     'minibatch': self.minibatch,
                     'best_loss': self.best_loss}

        if (valid_mae < self.best_loss):
            self.best_loss = save_dict['best_loss'] = valid_mae
            logging.info('Lowest loss achieved! Saving best result to file: {}'.format(self.args.bestfile))
            torch.save(save_dict, self.args.bestfile)

        logging.info('Saving to checkpoint file: {}'.format(self.args.checkfile))
        torch.save(save_dict, self.args.checkfile)

    def load_checkpoint(self):
        """
        Load checkpoint from previous file.
        """
        if not self.args.load:
            return
        elif os.path.exists(self.args.checkfile):
            logging.info('Loading previous model from checkpoint!')
            self.load_state(self.args.checkfile)
        else:
            logging.info('No checkpoint included! Starting fresh training program.')
            return

    def load_state(self, checkfile):
        logging.info('Loading from checkpoint!')

        checkpoint = torch.load(checkfile)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.minibatch = checkpoint['minibatch']

        logging.info('Best loss from checkpoint: {} at epoch {}'.format(self.best_loss, self.epoch))

    def evaluate(self, splits=['train', 'valid', 'test'], best=True, final=True):
        """
        Evaluate model on training/validation/testing splits.

        :splits: List of splits to include. Only valid splits are: 'train', 'valid', 'test'
        :best: Evaluate best model as determined by minimum validation error over evolution
        :final: Evaluate final model at end of training phase
        """
        if not self.args.save:
            logging.info('No model saved! Cannot give final status.')
            return

        # Evaluate final model (at end of training)
        if final:
            logging.info('Getting predictions for model in last checkpoint.')

            # Load checkpoint model to make predictions
            checkpoint = torch.load(self.args.checkfile)
            self.model.load_state_dict(checkpoint['model_state'])

            # Loop over splits, predict, and output/log predictions
            for split in splits:
                predict, targets = self.predict(split)
                self.log_predict(predict, targets, split, description='Final')

        # Evaluate best model as determined by validation error
        if best:
            logging.info('Getting predictions for best model.')

            # Load best model to make predictions
            checkpoint = torch.load(self.args.bestfile)
            self.model.load_state_dict(checkpoint['model_state'])

            # Loop over splits, predict, and output/log predictions
            for split in splits:
                predict, targets = self.predict(split)
                self.log_predict(predict, targets, split, description='Best')


        logging.info('Inference phase complete!')


    def _warm_restart(self, epoch):
        restart_epochs = self.restart_epochs

        if epoch in restart_epochs:
            logging.info('Warm learning rate restart at epoch {}!'.format(epoch))
            self.scheduler.last_epoch = 0
            idx = restart_epochs.index(epoch)
            self.scheduler.T_max = restart_epochs[idx+1] - restart_epochs[idx]
            if self.args.lr_minibatch:
                self.scheduler.T_max *= ceil(self.args.num_train / self.args.batch_size)
            self.scheduler.step(0)

    def _log_minibatch(self, batch_idx, loss, targets, predict, batch_t, epoch_t):
        mini_batch_loss = loss.item()
        mini_batch_mae = MAE(predict, targets)
        mini_batch_rmse = RMSE(predict, targets)

        # Exponential average of recent MAE/RMSE on training set for more convenient logging.
        if batch_idx == 0:
            self.mae, self.rmse = mini_batch_mae, mini_batch_rmse
        else:
            alpha = self.args.alpha
            self.mae = alpha * self.mae + (1 - alpha) * mini_batch_mae
            self.rmse = alpha * self.rmse + (1 - alpha) * mini_batch_rmse

        dtb = (datetime.now() - batch_t).total_seconds()
        tepoch = (datetime.now() - epoch_t).total_seconds()
        self.batch_time += dtb
        tcollate = tepoch-self.batch_time

        if self.args.textlog:
            logstring = 'E:{:3}/{}, B: {:5}/{}'.format(self.epoch+1, self.args.num_epoch, batch_idx, len(self.dataloaders['train']))
            logstring += '{:> 9.4f}{:> 9.4f}{:> 9.4f}'.format(sqrt(mini_batch_loss), self.mae, self.rmse)
            logstring += '  dt:{:> 6.2f}{:> 8.2f}{:> 8.2f}'.format(dtb, tepoch, tcollate)

            logging.info(logstring)

        if self.summarize:
            self.summarize.add_scalar('train/mae', sqrt(mini_batch_loss), self.minibatch)


    def _step_lr_batch(self):
        if self.args.lr_minibatch:
            self.scheduler.step()

    def _step_lr_epoch(self):
        if not self.args.lr_minibatch:
            self.scheduler.step()

    def train(self):
        epoch0 = self.epoch
        for epoch in range(epoch0, self.args.num_epoch):
            self.epoch = epoch
            epoch_time = datetime.now()
            logging.info('Starting Epoch: {}'.format(epoch+1))

            self._warm_restart(epoch)
            self._step_lr_epoch()

            train_predict, train_targets = self.train_epoch()
            valid_predict, valid_targets = self.predict('valid')

            train_mae, train_rmse = self.log_predict(train_predict, train_targets, 'train', epoch=epoch)
            valid_mae, valid_rmse = self.log_predict(valid_predict, valid_targets, 'valid', epoch=epoch)

            self._save_checkpoint(valid_mae)

            logging.info('Epoch {} complete!'.format(epoch+1))

    def _get_target(self, data, stats=None):
        """
        Get the learning target.
        If a stats dictionary is included, return a normalized learning target.
        """
        targets = data[self.args.target].to(self.device, self.dtype)

        if stats is not None:
            mu, sigma = stats[self.args.target]
            targets = (targets - mu) / sigma

        return targets

    def train_epoch(self):
        dataloader = self.dataloaders['train']

        current_idx, num_data_pts = 0, len(dataloader.dataset)
        self.mae, self.rmse, self.batch_time = 0, 0, 0
        all_predict, all_targets = [], []

        self.model.train()
        epoch_t = datetime.now()
        for batch_idx, data in enumerate(dataloader):
            batch_t = datetime.now()

            # Standard zero-gradient
            self.optimizer.zero_grad()

            # Get targets and predictions
            targets = self._get_target(data, self.stats)
            predict = self.model(data)

            # Calculate loss and backprop
            loss = self.loss_fn(predict, targets)
            loss.backward()

            # Step optimizer and learning rate
            self.optimizer.step()
            self._step_lr_batch()

            targets, predict = targets.detach().cpu(), predict.detach().cpu()

            all_predict.append(predict)
            all_targets.append(targets)

            self._log_minibatch(batch_idx, loss, targets, predict, batch_t, epoch_t)

            self.minibatch += 1

        all_predict = torch.cat(all_predict)
        all_targets = torch.cat(all_targets)

        return all_predict, all_targets

    def predict(self, set='valid'):
        dataloader = self.dataloaders[set]

        self.model.eval()
        all_predict, all_targets = [], []
        start_time = datetime.now()
        logging.info('Starting testing on {} set: '.format(set))

        for batch_idx, data in enumerate(dataloader):

            targets = self._get_target(data, self.stats)
            predict = self.model(data).detach()

            all_targets.append(targets)
            all_predict.append(predict)

        all_predict = torch.cat(all_predict)
        all_targets = torch.cat(all_targets)

        dt = (datetime.now() - start_time).total_seconds()
        logging.info(' Done! (Time: {}s)'.format(dt))

        return all_predict, all_targets

    def log_predict(self, predict, targets, dataset, epoch=-1, description='Current'):
        predict = predict.cpu().double()
        targets = targets.cpu().double()

        mae = MAE(predict, targets)
        rmse = RMSE(predict, targets)

        mu, sigma = self.stats[self.args.target]
        mae_units = sigma*mae
        rmse_units = sigma*rmse

        datastrings = {'train': 'Training', 'test': 'Testing', 'valid': 'Validation'}

        if epoch >= 0:
            suffix = 'final'
            logging.info('Epoch: {} Complete! {} {} Loss: {:10.4f} {:10.4f} Units: {:10.4f} {:10.4f}'.format(epoch+1, description, datastrings[dataset], mae, rmse, mae_units, rmse_units))
        else:
            suffix = 'best'
            logging.info('Training Complete! {} {} Loss: {:10.4f} {:10.4f} Units: {:10.4f} {:10.4f}'.format(description, datastrings[dataset], mae, rmse, mae_units, rmse_units))

        if self.args.predict:
            file = self.args.predictfile + '.' + suffix + '.' + dataset + '.pt'
            logging.info('Saving predictions to file: {}'.format(file))
            torch.save({'predict': predict, 'targets': targets}, file)

        return mae, rmse
