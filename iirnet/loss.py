import torch
import torch.fft
import numpy as np
import iirnet.signal as signal

def phase_loss(input_phs, target_phs):
    """Calculate phase loss accounting for phase wrapping at ±π"""
    # Find shortest distance between angles on unit circle
    phase_diff = torch.remainder(input_phs - target_phs + np.pi, 2*np.pi) - np.pi
    return torch.mean(phase_diff**2)

class LogMagFrequencyLoss(torch.nn.Module):
    def __init__(self, priority=False):
        super(LogMagFrequencyLoss, self).__init__()
        self.priority = priority

    def forward(self, input, target, eps=1e-8):
        bs = input.size(0)
        loss = 0

        if False:
            for n in range(bs):
                w, input_h = signal.sosfreqz(input[n, ...])
                w, target_h = signal.sosfreqz(target[n, ...])

                input_mag = 20 * torch.log10(signal.mag(input_h) + eps)
                target_mag = 20 * torch.log10(signal.mag(target_h) + eps)

                loss += torch.nn.functional.l1_loss(input_mag, target_mag)
        elif self.priority:
            # in this case, we compute the loss comparing the response as we increase the number
            # of biquads in the cascade, this should encourage to use lower order filter.

            # first compute the target response
            w, target_h = signal.sosfreqz(target, log=False)
            target_mag = 20 * torch.log10(signal.mag(target_h) + eps)

            n_sections = input.shape[1]
            mag_loss = 0
            # now compute error with each group of biquads
            for n in np.arange(n_sections, step=2):

                sos = input[:, 0 : n + 2, :]
                w, input_h = signal.sosfreqz(sos, log=False)
                input_mag = 20 * torch.log10(signal.mag(input_h) + eps)

                mag_loss += torch.nn.functional.mse_loss(input_mag, target_mag)

        else:
            w, input_h = signal.sosfreqz(input, log=False)
            w, target_h = signal.sosfreqz(target, log=False)

            input_mag = 20 * torch.log10(signal.mag(input_h) + eps)
            target_mag = 20 * torch.log10(signal.mag(target_h) + eps)

            mag_loss = torch.nn.functional.mse_loss(input_mag, target_mag)

        return mag_loss

class LogPhaseLoss(torch.nn.Module):
    def __init__(self, priority=False):
        super(LogPhaseLoss, self).__init__()
        self.priority = priority
    
    def forward(self, input, target, eps=1e-8, return_components=False):
        # Get frequency responses
        w, input_h = signal.sosfreqz(input, log=False)
        w, target_h = signal.sosfreqz(target, log=False)
        
        # Extract phase angle in radians
        input_phs = torch.angle(input_h)
        target_phs = torch.angle(target_h)
        
        # Use the unwrapped phase loss calculation
        phase_loss_val = phase_loss(input_phs, target_phs)
        
        # Also calculate magnitude metrics for reference/reporting
        input_mag = 20 * torch.log10(signal.mag(input_h) + eps)
        target_mag = 20 * torch.log10(signal.mag(target_h) + eps)
        mag_loss = torch.nn.functional.mse_loss(input_mag, target_mag)
        
        if return_components:
            return phase_loss_val, (mag_loss, phase_loss_val)
        return phase_loss_val


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
            phs_loss = phase_loss(input_phs, target_phs)
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
