import torch
import random
import pytorch_lightning as pl
from argparse import ArgumentParser

import iirnet.loss as loss
import iirnet.plotting as plotting
import iirnet.signal as signal


class IIRNet(pl.LightningModule):
    """Base IIRNet module."""

    def __init__(self, mag_weight=1.0, phase_weight=0.5, **kwargs):
        super(IIRNet, self).__init__()
        
        self.save_hyperparameters()
        
        # Update loss initialization with weights
        self.magfreqzloss = loss.FreqDomainLoss(
            mag_weight=self.hparams.mag_weight,
            phase_weight=self.hparams.phase_weight
        )
        self.dbmagfreqzloss = loss.LogMagFrequencyLoss(
            mag_weight=self.hparams.mag_weight,
            phase_weight=self.hparams.phase_weight
        )
        # Initialize lists to store validation metrics
        self.validation_step_mag_losses = []
        self.validation_step_phase_losses = []

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        mag_dB, mag_dB_norm, phs, real, imag, sos = batch
        pred_sos, _ = self(mag_dB_norm, phs)  # Forward pass
        # Get the loss and its raw components
        loss, (mag_loss, phs_loss) = self.dbmagfreqzloss(pred_sos, sos, return_components=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mag_loss", mag_loss, on_step=True, on_epoch=True)
        self.log("train_phase_loss", phs_loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        mag_dB, mag_dB_norm, phs, real, imag, sos = batch
        
        # Forward pass with both magnitude and phase
        pred_sos, zpk = self(mag_dB_norm, phs)
        
        # Get losses with components
        loss, (mag_loss, phs_loss) = self.dbmagfreqzloss(pred_sos, sos, return_components=True)

        # Log metrics without on_step to reduce overhead
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mag_loss", mag_loss, on_step=False, on_epoch=True)
        self.log("val_phase_loss", phs_loss, on_step=False, on_epoch=True)

        # move tensors to cpu for logging
        outputs = {
            "pred_sos": pred_sos.cpu(),
            "sos": sos.cpu(),
            "mag_dB": mag_dB.cpu(),
            "phs": phs.cpu(),  # Log the phase response
            "z": zpk[0].cpu(),
            "p": zpk[1].cpu(),
            "k": zpk[2].cpu(),
        }
        # Return outputs and losses for callbacks
        outputs.update({
            "val_loss": loss,
            "mag_loss": mag_loss,
            "phase_loss": phs_loss
        })

        return outputs

    def on_validation_epoch_start(self):
        # Clear lists at start of validation
        self.validation_step_mag_losses = []
        self.validation_step_phase_losses = []

    def on_validation_epoch_end(self):
        if len(self.validation_step_mag_losses) > 0:
            avg_mag_loss = torch.stack(self.validation_step_mag_losses).mean()
            avg_phase_loss = torch.stack(self.validation_step_phase_losses).mean()
            
            self.log("val_mag_loss_epoch", avg_mag_loss)
            self.log("val_phase_loss_epoch", avg_phase_loss)

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # ...existing arguments...
        parser.add_argument("--num_points", type=int, default=512)
        parser.add_argument("--num_layers", type=int, default=4)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--filter_order", type=int, default=2)
        parser.add_argument("--lr", type=float, default=1e-3)
        
        return parser