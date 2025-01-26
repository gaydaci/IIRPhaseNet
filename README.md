# project-7

# IIRNet Phase Extension

Directly extending IIRNet to include phase response optimization for better filter design performance.

---

## Overview
This project extends the foundational work from the [IIRNet paper](https://arxiv.org/abs/2110.03691) by incorporating phase response optimization. By addressing phase discontinuities and unwrapping, the enhanced IIRNet enables robust magnitude and phase predictions.

---

## Key Features

### Data Pipeline
- **Extended Data Points:** Frequency range increased to 1024 points.
- **Phase Response Support:** Full handling and representation of phase data.

### Loss Functions
- **Phase-Aware Loss:** Introduced loss terms for phase unwrapping and balancing.
- **Optimized Learning:** Magnitude and phase contributions are balanced.

### Training Setup
- **Custom Colab Notebook:** Simplified interface for training, logging, and visualization. [Try it here](https://colab.research.google.com/drive/1Jolw2vDQuo0soO_ZCpvSyHBc8SP-XPNE#scrollTo=HVHMZq3JtJTN&uniqifier=1).
- **Batch Configurations:** Defaults include 1000 training and 100 validation samples.
- **Model Settings:** Adjustable parameters for filter method, order, and training epochs.

### Architecture Updates
- **Input Layer Expansion:** Doubled to support 1024 points.
- **Model Refinements:** Hidden layers optimized for phase data integration.

---

## Simple Training Workflow

### Steps:
1. **Setup & Imports:** Load PyTorch, Lightning, and custom modules.
2. **Argument Parsing:** Configure training settings (e.g., batch size, epochs).
3. **Training Pipeline:** Automatically handles logging, progress tracking, and model summaries.
4. **Data Handling:** Prepare IIRFilter datasets for training and validation.
5. **Run Training:** Execute training loop with full TensorBoard integration.

---

## Next Steps

1. **Phase Optimization**
   - Advanced handling for discontinuities and unwrapping.
   - Refining loss contributions for magnitude/phase balance.

2. **Architecture Refinements**
   - Experiment with LSTM-based models for sequence learning.
   - Hyperparameter tuning to improve generalization.

3. **Validation**
   - Compare against HRTF datasets ([RIEC HRTF](https://www.firsuite.net/)) and industry methods.

---

## Citation

If you use this work, please cite the original IIRNet paper:
```bibtex
@inproceedings{colonel2021iirnet,
  title={Direct design of biquad filter cascades with deep learning by sampling random polynomials},
  author={Colonel, Joseph and Steinmetz, Christian J. and Michelen, Marcus and Reiss, Joshua D.},
  booktitle={ICASSP},
  year={2022}
}
```