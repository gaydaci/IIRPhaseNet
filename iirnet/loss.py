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

    def forward(self, input, target, eps=1e-8):
        if self.priority:
            return self._priority_forward(input, target, eps)
        else:
            return self._standard_forward(input, target, eps)

    def _standard_forward(self, input, target, eps=1e-8):
        w, input_h = signal.sosfreqz(input, log=False)
        w, target_h = signal.sosfreqz(target, log=False)

        # Magnitude processing
        input_mag = 20 * torch.log10(signal.mag(input_h) + eps)
        target_mag = 20 * torch.log10(signal.mag(target_h) + eps)
        
        # Phase processing
        input_phs = torch.angle(input_h)
        target_phs = torch.angle(target_h)

        # Frequency weighting
        if self.freq_dependent:
            freq_weights = 1.0 / (1.0 + w/np.pi)
        else:
            freq_weights = torch.ones_like(w)

        # Weighted losses
        mag_loss = torch.mean(freq_weights * (input_mag - target_mag)**2)
        phs_loss = torch.mean(freq_weights * (input_phs - target_phs)**2)

        return self.mag_weight * mag_loss + self.phase_weight * phs_loss

    def _priority_forward(self, input, target, eps=1e-8):
        # Keep existing priority implementation
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
                freq_weights = 1.0 / (1.0 + w/np.pi)
                mag_loss += torch.mean(freq_weights * (input_mag - target_mag)**2)
                phs_loss += torch.mean(freq_weights * (input_phs - target_phs)**2)
            else:
                mag_loss += torch.nn.functional.mse_loss(input_mag, target_mag)
                phs_loss += torch.nn.functional.mse_loss(input_phs, target_phs)

        return self.mag_weight * mag_loss + self.phase_weight * phs_loss


class FreqDomainLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input,
        target,
        eps=1e-8,
        error="l2",
    ):

        _, input_h = signal.sosfreqz(input, log=False)
        _, target_h = signal.sosfreqz(target, log=False)

        input_mag = signal.mag(input_h)
        target_mag = signal.mag(target_h)
        input_phs = torch.angle(input_h)
        target_phs = torch.angle(target_h)

        input_mag_log = torch.log(input_mag)
        target_mag_log = torch.log(target_mag)

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
    def __init__(self, priority=False, use_dB=True, zero_mean=False):
        super(LogMagTargetFrequencyLoss, self).__init__()
        self.priority = priority
        self.use_dB = use_dB
        self.zero_mean = zero_mean

    def forward(self, input_sos, target_mag, eps=1e-8):

        w, input_h = signal.sosfreqz(input_sos, worN=target_mag.shape[-1], log=False)
        input_mag = 20 * torch.log10(signal.mag(input_h) + eps).float()
        input_phs = torch.angle(input_h)

        mag_loss = torch.nn.functional.mse_loss(input_mag, target_mag)
        phs_loss = torch.nn.functional.mse_loss(input_phs, target_mag)

        return mag_loss + phs_loss


class ComplexLoss(torch.nn.Module):
    def __init__(self, threshold=1e-16):
        super(ComplexLoss, self).__init__()
        self.threshold = threshold

    def forward(self, input, target):
        bs = input.size(0)
        loss = 0

        if False:
            for n in range(bs):

                input_sos = input[n, ...]
                target_sos = target[n, ...]

                if self.threshold is not None:
                    input_sos = self.apply_threshold(input_sos)
                    target_sos = self.apply_threshold(target_sos)

                w, input_h = signal.sosfreqz(input_sos, log=True)
                w, target_h = signal.sosfreqz(target_sos, log=True)

                real_loss = torch.nn.functional.l1_loss(input_h.real, target_h.real)
                imag_loss = torch.nn.functional.l1_loss(input_h.imag, target_h.imag)
                loss += real_loss + imag_loss
        else:
            w, input_h = signal.sosfreqz(input, log=False)
            w, target_h = signal.sosfreqz(target, log=False)
            real_loss = torch.nn.functional.mse_loss(input_h.real, target_h.real)
            imag_loss = torch.nn.functional.mse_loss(input_h.imag, target_h.imag)
            phs_loss = torch.nn.functional.mse_loss(torch.angle(input_h), torch.angle(target_h))
            loss = real_loss + imag_loss + phs_loss

        return torch.mean(loss)

    def apply_threshold(self, sos):
        out_sos = sos[sos.sum(-1) > self.threshold, :]

        # check if all sections were removed
        if out_sos.size(0) == 0:
            out_sos = sos

        return out_sos
