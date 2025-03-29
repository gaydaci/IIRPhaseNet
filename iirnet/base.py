import torch
import random
import pytorch_lightning as pl
from argparse import ArgumentParser

import iirnet.loss as loss
import iirnet.plotting as plotting
import iirnet.signal as signal


class IIRNet(pl.LightningModule):
    """Base IIRNet module."""

    def __init__(self, mag_weight=1.0, phase_weight=0.5, use_complex_loss=False, **kwargs):
        super(IIRNet, self).__init__()
        
        self.save_hyperparameters()
        
        # Initialize appropriate loss function based on config
        if self.hparams.use_complex_loss:
            print("Using complex plane optimization loss (no manual weights needed)")
            self.dbmagfreqzloss = loss.ComplexPlaneOptimizationLoss(
                log_domain=True,
                normalize=True
            )
            self.magfreqzloss = loss.ComplexPlaneOptimizationLoss(
                log_domain=False,
                normalize=True
            )
        else:
            print(f"Using weighted loss with mag_weight={self.hparams.mag_weight}, phase_weight={self.hparams.phase_weight}")
            self.dbmagfreqzloss = loss.LogMagFrequencyLoss(
                mag_weight=self.hparams.mag_weight,
                phase_weight=self.hparams.phase_weight
            )
            self.magfreqzloss = loss.FreqDomainLoss(
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
        pred_sos, _ = self(mag_dB_norm, phs)
        
        # Get both the loss and components
        if self.hparams.use_complex_loss:
            loss, (mag_loss, phs_loss) = self.magfreqzloss(pred_sos, sos, return_components=True)
        else:
            loss = self.magfreqzloss(pred_sos, sos)
            _, (mag_loss, phs_loss) = self.magfreqzloss(pred_sos, sos, return_components=True)
        
        # Log with better detail
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mag_loss", mag_loss, on_step=False, on_epoch=True)
        self.log("train_phase_loss", phs_loss, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        mag_dB, mag_dB_norm, phs, real, imag, sos = batch
        # Forward pass using magnitude and phase inputs
        pred_sos, zpk = self(mag_dB_norm, phs)
        
        # Get both the overall loss and component losses
        if self.hparams.use_complex_loss:
            # For complex loss, get the components explicitly
            loss, (mag_loss, phs_loss) = self.magfreqzloss(pred_sos, sos, return_components=True)
            dB_MSE, _ = self.dbmagfreqzloss(pred_sos, sos, return_components=True)
        else:
            # For weighted loss
            loss = self.magfreqzloss(pred_sos, sos)
            dB_MSE = self.dbmagfreqzloss(pred_sos, sos)
            
            # Extract components for consistent reporting
            _, (mag_loss, phs_loss) = self.magfreqzloss(pred_sos, sos, return_components=True)
        
        # Store components for epoch-end statistics
        self.validation_step_mag_losses.append(mag_loss)
        self.validation_step_phase_losses.append(phs_loss)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("dB_MSE", dB_MSE, on_step=False, on_epoch=True)
        self.log("val_mag_loss_step", mag_loss, on_step=True, on_epoch=False)
        self.log("val_phase_loss_step", phs_loss, on_step=True, on_epoch=False)
        
        outputs = {
            "pred_sos": pred_sos.cpu(),
            "sos": sos.cpu(),
            "mag_dB": mag_dB.cpu(),
            "z": zpk[0].cpu(),
            "p": zpk[1].cpu(),
            "k": zpk[2].cpu(),
        }
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
        # new argument for complex loss
        parser.add_argument("--use_complex_loss", action="store_true",
                            help="Use complex plane optimization instead of weighted mag/phase")       
        
        return parser