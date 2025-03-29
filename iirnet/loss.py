import torch
import torch.fft
import numpy as np
import iirnet.signal as signal

class LogMagFrequencyLoss(torch.nn.Module):
    def __init__(self, mag_weight=1.0, phase_weight=0.5):
        super(LogMagFrequencyLoss, self).__init__()
        self.mag_weight = mag_weight
        self.phase_weight = phase_weight

    def forward(self, input, target, eps=1e-8, return_components=False):
        # Use the same frequency response settings as original
        w, input_h = signal.sosfreqz(input, log=False)
        w, target_h = signal.sosfreqz(target, log=False)
        
        # Magnitude processing (dB-scale)
        input_mag = 20 * torch.log10(signal.mag(input_h) + eps)
        target_mag = 20 * torch.log10(signal.mag(target_h) + eps)
        
        # If phase loss is off, bypass phase computation entirely
        if self.phase_weight == 0:
            mag_loss = torch.nn.functional.mse_loss(input_mag, target_mag)
            loss = self.mag_weight * mag_loss
            if return_components:
                return loss, (self.mag_weight * mag_loss, torch.tensor(0.0, device=loss.device))
            return loss
        
        # Otherwise, compute phase responses and loss
        input_phs = torch.angle(input_h)
        target_phs = torch.angle(target_h)
        
        mag_loss = torch.nn.functional.mse_loss(input_mag, target_mag)
        phs_loss = torch.nn.functional.mse_loss(input_phs, target_phs)
        
        loss = self.mag_weight * mag_loss + self.phase_weight * phs_loss
        
        if return_components:
            return loss, (self.mag_weight * mag_loss, self.phase_weight * phs_loss)
        return loss


class FreqDomainLoss(torch.nn.Module):
    def __init__(self, mag_weight=1.0, phase_weight=0.5):
        super(FreqDomainLoss, self).__init__()
        self.mag_weight = mag_weight
        self.phase_weight = phase_weight

    def forward(self, input, target, eps=1e-8, error="l2", return_components=False):
        _, input_h = signal.sosfreqz(input, log=False)
        _, target_h = signal.sosfreqz(target, log=False)

        input_mag = signal.mag(input_h)
        target_mag = signal.mag(target_h)
        input_phs = torch.angle(input_h)
        target_phs = torch.angle(target_h)

        input_mag_log = torch.log(input_mag + eps)
        target_mag_log = torch.log(target_mag + eps)

        if error == "l1":
            mag_log_loss = torch.nn.functional.l1_loss(input_mag_log, target_mag_log)
            phs_loss = torch.nn.functional.l1_loss(input_phs, target_phs)
        elif error == "l2":
            mag_log_loss = torch.nn.functional.mse_loss(input_mag_log, target_mag_log)
            phs_loss = torch.nn.functional.mse_loss(input_phs, target_phs)
        else:
            raise RuntimeError(f"Invalid `error`: {error}.")

        # Apply weights
        final_loss = self.mag_weight * mag_log_loss + self.phase_weight * phs_loss
        
        # Return components if requested
        if return_components:
            return final_loss, (mag_log_loss, phs_loss)
        return final_loss


class LogMagTargetFrequencyLoss(torch.nn.Module):
    def __init__(self, mag_weight=1.0, phase_weight=0.5):
        super(LogMagTargetFrequencyLoss, self).__init__()
        self.mag_weight = mag_weight
        self.phase_weight = phase_weight

    def forward(self, input_sos, target_mag, target_phs, eps=1e-8):
        w, input_h = signal.sosfreqz(input_sos, worN=target_mag.shape[-1], log=False)
        input_mag = 20 * torch.log10(signal.mag(input_h) + eps).float()
        input_phs = torch.angle(input_h)

        mag_loss = torch.nn.functional.mse_loss(input_mag, target_mag)
        phs_loss = torch.nn.functional.mse_loss(input_phs, target_phs)
        
        # Apply weights
        final_loss = self.mag_weight * mag_loss + self.phase_weight * phs_loss
        
        print(f"DEBUG (TargetFreq): mag_loss={mag_loss:.6f}, phs_loss={phs_loss:.6f}")
        
        return final_loss


class ComplexLoss(torch.nn.Module):
    def __init__(self, mag_weight=1.0, phase_weight=0.5, threshold=1e-16):
        super(ComplexLoss, self).__init__()
        self.mag_weight = mag_weight
        self.phase_weight = phase_weight
        self.threshold = threshold

    def forward(self, input, target):
        _, input_h = signal.sosfreqz(input, log=False)
        _, target_h = signal.sosfreqz(target, log=False)
        
        real_loss = torch.nn.functional.mse_loss(input_h.real, target_h.real)
        imag_loss = torch.nn.functional.mse_loss(input_h.imag, target_h.imag)
        phs_loss = torch.nn.functional.mse_loss(torch.angle(input_h), torch.angle(target_h))
        
        # Apply weights
        final_loss = self.mag_weight * (real_loss + imag_loss) + self.phase_weight * phs_loss
        
        return final_loss

    def apply_threshold(self, sos):
        out_sos = sos[sos.sum(-1) > self.threshold, :]
        if out_sos.size(0) == 0:
            out_sos = sos
        return out_sos

class ComplexPlaneOptimizationLoss(torch.nn.Module):
    def __init__(self, log_domain=True, normalize=True):
        super(ComplexPlaneOptimizationLoss, self).__init__()
        self.log_domain = log_domain
        self.normalize = normalize
        
    def forward(self, input, target, eps=1e-8, return_components=False):
        # Get frequency responses
        w, input_h = signal.sosfreqz(input, worN=512)
        w, target_h = signal.sosfreqz(target, worN=512)
        
        # For reporting: extract magnitude and phase components
        input_mag = 20 * torch.log10(signal.mag(input_h) + eps)
        target_mag = 20 * torch.log10(signal.mag(target_h) + eps)
        input_phs = torch.angle(input_h)
        target_phs = torch.angle(target_h)
        
        if self.log_domain:
            # Log-complex domain
            input_log = torch.log(input_h + eps)
            target_log = torch.log(target_h + eps)
            
            # Split real and imaginary parts for MSE calculation
            real_loss = torch.nn.functional.mse_loss(input_log.real, target_log.real)
            imag_loss = torch.nn.functional.mse_loss(input_log.imag, target_log.imag)
            complex_loss = real_loss + imag_loss
        else:
            # Direct complex domain
            if self.normalize:
                # Normalize to balance contributions
                scale = torch.mean(torch.abs(target_h)) + eps
                input_norm = input_h / scale
                target_norm = target_h / scale
            else:
                input_norm = input_h
                target_norm = target_h
                
            # Split real and imaginary parts for MSE calculation
            real_loss = torch.nn.functional.mse_loss(input_norm.real, target_norm.real)
            imag_loss = torch.nn.functional.mse_loss(input_norm.imag, target_norm.imag)
            complex_loss = real_loss + imag_loss
        
        # Calculate separate mag/phase losses for reporting
        mag_loss = torch.nn.functional.mse_loss(input_mag, target_mag)
        phs_loss = torch.nn.functional.mse_loss(input_phs, target_phs)
        
        if return_components:
            return complex_loss, (mag_loss, phs_loss)
        return complex_loss