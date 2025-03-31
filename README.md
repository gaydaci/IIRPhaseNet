# IIR Phase Net

IIR Phase Net extends the original IIRNet framework by incorporating phase information into both the input representation and the loss function. This allows the model to design stable IIR filters that match both magnitude and phase responses—even for non-minimum-phase filters.

## Overview

- **Dual-Domain Input:**  
  The network accepts a concatenated vector of log-magnitude and wrapped phase spectra (e.g., 512 bins each by default).

- **Phase-Aware Loss:**  
  Uses a composite loss that combines the mean squared error (MSE) on logarithmic magnitude and a wrapped angular distance metric for phase. You can adjust the balance between magnitude and phase errors via tunable weights.

- **Max Phase Design:**  
  Enforces filter stability by reparameterizing predicted poles to lie inside the unit circle while allowing zeros to be free, thereby handling non-minimum-phase responses.

- **Flexible Architecture:**  
  Experiments have been conducted with different network depths (from 2 to 8 hidden layers) to find the best trade-off between model capacity and training stability.

### Train Your Own Model

1. **Data Preparation:**  
   IIR Phase Net trains on a diverse dataset of random IIR filters generated from several polynomial families (e.g., `normal_poly`, `uniform_mag_disk`, `char_poly`).

2. **Training Setup:**  
   Use the provided training scripts. For example:

   ```bash
   ./configs/train_hidden_dim.sh
   ./configs/filter_method.sh
   ./configs/filter_order.sh
   ```

   These scripts launch training jobs with different settings (e.g., network depth, loss weightings) and log metrics such as training loss, decibel MSE, and phase error.

3. **Monitor Training:**  
   Launch TensorBoard to view training progress:

   ```bash
   tensorboard --logdir=logs
   ```


## Additional Command-Line Options

You can adjust various parameters such as:
- **Loss Weights:** Set `mag_weight` and `phase_weight` to balance magnitude and phase errors.
- **Network Depth:** Adjust the number of hidden layers (e.g., `--num_layers 4`).
- **Input Resolution:** Change the number of frequency points (e.g., `--num_points 512`).

## Citation

If you use IIR Net in your work, please cite:

> J. T. Colonel, C. J. Steinmetz, M. Michelen, and J. D. Reiss, "Direct Design of Biquad Filter Cascades with Deep Learning by Sampling Random Polynomials," Proc. IEEE ICASSP, 2022.