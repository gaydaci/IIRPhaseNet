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
        self.magfreqzloss = loss.FreqDomainLoss()
        self.dbmagfreqzloss = loss.LogMagFrequencyLoss(
            mag_weight=self.hparams.mag_weight,
            phase_weight=self.hparams.phase_weight
        )

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
        mag_dB, mag_dB_norm, phs, real, imag, sos = batch
        pred_sos, zpk = self(mag_dB_norm, phs)
        loss, (mag_loss, phs_loss) = self.dbmagfreqzloss(pred_sos, sos, return_components=True)
        
        # Debug validation losses
        print(f"VAL DEBUG: Step {batch_idx}")
        print(f"VAL DEBUG: Raw mag_loss={mag_loss.item():.6f}")
        print(f"VAL DEBUG: Raw phs_loss={phs_loss.item():.6f}")
        
        # Log with sync_dist for proper multi-GPU handling
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_mag_loss", mag_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_phase_loss", phs_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        # Add per-step logging for debugging
        self.log("val_mag_loss_step", mag_loss, on_step=True, on_epoch=False)
        self.log("val_phase_loss_step", phs_loss, on_step=True, on_epoch=False)

        outputs = {
            "pred_sos": pred_sos.cpu(),
            "sos": sos.cpu(),
            "mag_dB": mag_dB.cpu(),
            "phs": phs.cpu(),
            "z": zpk[0].cpu(),
            "p": zpk[1].cpu(),
            "k": zpk[2].cpu(),
        }
        return outputs

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
