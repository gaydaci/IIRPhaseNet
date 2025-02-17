import torch
import torch.fft
import numpy as np
import iirnet.signal as signal

class LogMagFrequencyLoss(torch.nn.Module):
    def __init__(self, mag_weight=1.0, phase_weight=0.5, priority=False, freq_dependent=True):
        super(LogMagFrequencyLoss, self).__init__()
        self.mag_weight = mag_weight
        self.phase_weight = phase_weight
        self.priority = priority
        self.freq_dependent = freq_dependent

    def forward(self, input, target, eps=1e-8, return_components=False):
        if self.priority:
            loss = self._priority_forward(input, target, eps, return_components)
        else:
            loss = self._standard_forward(input, target, eps, return_components)
        return loss

    def _standard_forward(self, input, target, eps=1e-8, return_components=False):
        # Match points to input size
        w, input_h = signal.sosfreqz(input, worN=input.shape[-1])
        w, target_h = signal.sosfreqz(target, worN=input.shape[-1])
        
        # Magnitude processing
        input_mag = 20 * torch.log10(signal.mag(input_h) + eps)
        target_mag = 20 * torch.log10(signal.mag(target_h) + eps)
        
        # Phase processing - Fix gradient issue
        input_phs = torch.angle(input_h)
        target_phs = torch.angle(target_h)
        
        # Unwrap phase while preserving gradients
        with torch.no_grad():
            input_phs_np = input_phs.detach().cpu().numpy()
            target_phs_np = target_phs.detach().cpu().numpy()
            input_phs_unwrap = np.unwrap(input_phs_np)
            target_phs_unwrap = np.unwrap(target_phs_np)
        
        # Convert back to tensors and preserve gradients
        input_phs_unwrap = torch.tensor(input_phs_unwrap, device=input.device, dtype=input.dtype)
        target_phs_unwrap = torch.tensor(target_phs_unwrap, device=target.device, dtype=target.dtype)
        input_phs_unwrap.requires_grad_(True)
        
        # Compute normalized phase difference
        phase_diff = input_phs_unwrap - target_phs_unwrap
        
        # Compute losses
        mag_loss = torch.mean((input_mag - target_mag)**2) / (128.0**2)
        phs_loss = torch.mean(phase_diff**2) / (np.pi**2)
        
        # Apply weights
        final_loss = self.mag_weight * mag_loss + self.phase_weight * phs_loss
        
        # Debug prints
        print(f"DEBUG: mag_w={self.mag_weight:.2f}, phs_w={self.phase_weight:.2f}")
        print(f"DEBUG: mag_loss={mag_loss:.6f}, phs_loss={phs_loss:.6f}")
        
        if return_components:
            return final_loss, (mag_loss, phs_loss)
        return final_loss

    def _priority_forward(self, input, target, eps=1e-8, return_components=False):
        # Priority mode: accumulate loss over subsections of the filter
        w, target_h = signal.sosfreqz(target, log=False)
        target_mag = 20 * torch.log10(signal.mag(target_h) + eps)
        target_phs = torch.angle(target_h)

        n_sections = input.shape[1]
        mag_loss = 0
        phs_loss = 0

        for n in np.arange(n_sections, step=2):
            sos = input[:, 0 : n + 2, :]
            w, input_h = signal.sosfreqz(sos, log=False)
            input_mag = 20 * torch.log10(signal.mag(input_h) + eps)
            input_phs = torch.angle(input_h)

            if self.freq_dependent:
                freq_weights = 1.0 / (1.0 + w / np.pi)
                freq_weights = torch.from_numpy(freq_weights).to(input_mag.device)
                mag_loss += torch.mean(freq_weights * (input_mag - target_mag) ** 2)
                phs_loss += torch.mean(freq_weights * (input_phs - target_phs) ** 2)
            else:
                mag_loss += torch.nn.functional.mse_loss(input_mag, target_mag)
                phs_loss += torch.nn.functional.mse_loss(input_phs, target_phs)

        final_loss = self.mag_weight * mag_loss + self.phase_weight * phs_loss
        if return_components:
            return final_loss, (mag_loss, phs_loss)
        else:
            return final_loss


class FreqDomainLoss(torch.nn.Module):
    def __init__(self):
        super(FreqDomainLoss, self).__init__()

    def forward(self, input, target, eps=1e-8, error="l2"):
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

        return mag_log_loss + phs_loss


class LogMagTargetFrequencyLoss(torch.nn.Module):
    """
    Compare the predicted frequency response (both magnitude and phase) with provided target values.
    Note: The forward method now requires both target_mag and target_phs.
    """
    def __init__(self, priority=False, use_dB=True, zero_mean=False):
        super(LogMagTargetFrequencyLoss, self).__init__()
        self.priority = priority
        self.use_dB = use_dB
        self.zero_mean = zero_mean

    def forward(self, input_sos, target_mag, target_phs, eps=1e-8):
        w, input_h = signal.sosfreqz(input_sos, worN=target_mag.shape[-1], log=False)
        input_mag = 20 * torch.log10(signal.mag(input_h) + eps).float()
        input_phs = torch.angle(input_h)

        mag_loss = torch.nn.functional.mse_loss(input_mag, target_mag)
        phs_loss = torch.nn.functional.mse_loss(input_phs, target_phs)
        print("DEBUG (TargetFreq): mag_loss={:.6f}, phs_loss={:.6f}".format(mag_loss.item(), phs_loss.item()))
        return mag_loss + phs_loss


class ComplexLoss(torch.nn.Module):
    def __init__(self, threshold=1e-16):
        super(ComplexLoss, self).__init__()
        self.threshold = threshold

    def forward(self, input, target):
        _, input_h = signal.sosfreqz(input, log=False)
        _, target_h = signal.sosfreqz(target, log=False)
        real_loss = torch.nn.functional.mse_loss(input_h.real, target_h.real)
        imag_loss = torch.nn.functional.mse_loss(input_h.imag, target_h.imag)
        phs_loss = torch.nn.functional.mse_loss(torch.angle(input_h), torch.angle(target_h))
        loss = real_loss + imag_loss + phs_loss
        return torch.mean(loss)

    def apply_threshold(self, sos):
        out_sos = sos[sos.sum(-1) > self.threshold, :]
        if out_sos.size(0) == 0:
            out_sos = sos
        return out_sos
