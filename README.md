# IIRNet Phase Extension

Directly extending IIRNet to optimize both magnitude and phase responses for more accurate filter design.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csteinmetz1/IIRNet/blob/main/demos/demo.ipynb)  
[![arXiv](https://img.shields.io/badge/arXiv-2110.03691-b31b1b.svg)](https://arxiv.org/abs/2110.03691)

<a href="https://www.youtube.com/watch?v=B_fRxHZMJ9k" target="_blank">
  <img width="650px" src="docs/assets/thumbnail.png" alt="Paper Video Explanation">
</a>

---

## Overview

This project builds on the original [IIRNet paper](https://arxiv.org/abs/2110.03691) by incorporating phase response optimization into the network. By addressing phase discontinuities and introducing phase-aware loss functions, the enhanced model is capable of predicting IIR filter coefficients that more accurately match both the magnitude and phase responses of the target filter. Our improvements include:

- **Phase Response Optimization:** New loss functions and training metrics account for phase unwrapping and balancing.
- **Extended Frequency Representation:** The data pipeline now supports an increased frequency range (up to 1024 points) to capture finer spectral details.
- **Architectural Refinements:** Updated network modules and an expanded input layer to integrate phase data, along with experiments using deeper architectures (8 hidden layers).
- **Evaluation Metrics:** In addition to overall training/validation losses, we report the decibel mean-squared error (dB_MSE) metric that evaluates the magnitude error, while separate phase error metrics (in radians) are also logged.

---

## Key Features

### Data Pipeline
- **Extended Data Points:** Frequency range increased to 1024 points.
- **Phase Response Support:** Both magnitude and phase information are now handled explicitly.

### Loss Functions & Metrics
- **Phase-Aware Loss:** Our customized loss functions combine a magnitude loss (computed in dB scale) and a phase loss (using an unwrapped phase error measure).  
- **Weighted Loss Strategy:** Magnitude and phase contributions can be tuned via `mag_weight` and `phase_weight` parameters. (For instance, when phase loss is much smaller, increasing its weight—e.g., `phase_weight=50`—can balance the overall loss.)
- **dB_MSE Metric:** A separate metric (based on the LogMagFrequencyLoss with phase logic removed) reports the mean-squared error on the magnitude response in decibels.
- **Phase Error Metric:** The network also logs phase error (in radians) for additional insight during evaluation.

### Training Setup
- **Custom Colab Notebook:** Simplified interface for training, logging, and visualization. [Try it here](https://colab.research.google.com/drive/1Jolw2vDQuo0soO_ZCpvSyHBc8SP-XPNE#scrollTo=HVHMZqJtJTN&uniqifier=1).
- **Batch Configurations:** Defaults include 1000 training and 100 validation samples.
- **Model Settings:** Adjustable parameters for filter family, filter order, network depth (e.g., 8 hidden layers), and learning epochs.
- **Optimizer & Scheduler:** Training is performed with AdamW and learning rate scheduling, with gradient clipping for stability.

### Architecture Updates
- **Input Layer Expansion:** The input layer now accepts 1024 frequency points.
- **Deeper Network:** Experiments with 8 hidden layers (and LSTM variants) improve the integration of phase data and overall model capacity.
- **Phase Integration:** Updated modules now process phase information alongside magnitude, while still optionally supporting a minimum phase constraint for stability.

---

## Training Workflow

### Setup & Installation
1. **Clone the Repository and Install Dependencies:**

    ```bash
    git clone https://github.com/csteinmetz1/IIRNet.git
    cd IIRNet
    pip install -e .
    ```

2. **Configure Training Settings:**  
   Use the provided argument parser to set options (e.g., `--num_layers 8`, `--max_train_order 100`, `--mag_weight 1.0`, `--phase_weight 0.5`, or other values to balance loss components).

### Running the Training
1. **Data Preparation:**  
   The IIRFilterDataset class handles random filter generation using diverse polynomial families, now with phase information.
   
2. **Training:**  
   Launch training using one of the provided shell scripts in the `configs/` directory:

    ```bash
    ./configs/train_hidden_dim.sh
    ./configs/filter_method.sh
    ./configs/filter_order.sh
    ```

3. **Logging:**  
   TensorBoard integration is enabled—view training progress and logged metrics (train_loss, val_loss, dB_MSE, phase error, etc.) with:

    ```bash
    tensorboard --logdir logs
    ```

### Evaluation
1. **Download Datasets:**  
   Navigate to the `data` directory and run:

    ```bash
    cd data
    ./dl.sh
    ```
   
2. **Pre-trained Checkpoints:**  
   Download the pre-trained models using the provided scripts and unzip them in the `logs` directory.

3. **Run Evaluation Scripts:**

    ```bash
    python eval.py logs/filter_method --yw --sgd --guitar_cab --hrtf --filter_order 16
    python eval.py logs/hidden_dim --yw --sgd --guitar_cab --hrtf --filter_order 16
    python eval.py logs/filter_order --guitar_cab --hrtf --filter_order 4
    python eval.py logs/filter_order --guitar_cab --hrtf --filter_order 8
    python eval.py logs/filter_order --guitar_cab --hrtf --filter_order 16
    python eval.py logs/filter_order --guitar_cab --hrtf --filter_order 32
    python eval.py logs/filter_order --guitar_cab --hrtf --filter_order 64
    ```

---

## Filter Design Example

Below is an example script that shows how to design a filter using the trained model:

```python
from iirnet.designer import Designer
import scipy.signal
import torch
import numpy as np
import matplotlib.pyplot as plt

n = 32  # Desired filter order (choose from 4, 8, 16, 32, 64)
m = [0, -3, 0, 12, 0, -6, 0]  # Target magnitude response specification
mode = "linear"  # Interpolation mode for specification
output = "sos"   # Output type ("sos" or "ba")

designer = Designer()
sos = designer(n, m, mode=mode, output=output)

w, h = scipy.signal.sosfreqz(sos.numpy(), fs=2)

# Interpolate target for plotting
m_int = torch.tensor(m).view(1, 1, -1).float()
m_int = torch.nn.functional.interpolate(m_int, 512, mode=mode)

fig, ax = plt.subplots(figsize=(6, 3))
plt.plot(w, 20 * np.log10(np.abs(h)), label="Estimation")
plt.plot(w, m_int.view(-1), label="Specification")
plt.legend()
plt.show()
