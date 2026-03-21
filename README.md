# DeepLense — Hybrid Quantum-Classical Representation Learning for Dark Matter Substructure Classification

**GSoC 2026 Evaluation Submission · ML4SCI · DeepLense**

| | |
|---|---|
| **Author** | Aditya Raj |
| **GitHub** | [github.com/Optimus2007](https://github.com/Optimus2007) |
| **LinkedIn** | [linkedin.com/in/aditya-raj-3605b233a](https://linkedin.com/in/aditya-raj-3605b233a/) |
| **Project** | Hybrid Quantum-Classical Representation Learning for Dark Matter Substructure |
| **Programme** | Google Summer of Code 2026 · ML4SCI · DeepLense |

---

## Overview

Strong gravitational lensing occurs when a massive foreground object bends light from a background source into arcs or rings. The exact shape of that ring encodes information about the **dark matter distribution** in the lens — specifically, whether it contains smooth halos, cold dark matter subhalos, or axion/fuzzy dark matter vortices. Classifying these three regimes from telescope images is a core problem in observational cosmology.

This repository contains two evaluation task submissions:

| File | Task | Approach |
|------|------|----------|
| `Common_Test.ipynb` | Common Task — Classical deep learning | Transfer learning: EfficientNet-B3, ResNet-50, ConvNeXt-Tiny |
| `Quantum_ML.ipynb` | Specific Task — Quantum ML | 6 quantum experiments from VQC to fine-tuned hybrid |

**Classification target:**

| Class | Folder | Physical Description |
|-------|--------|----------------------|
| No Substructure | `no` | Smooth Einstein ring — no dark matter substructure |
| Subhalo (CDM) | `sphere` | Cold Dark Matter subhalo perturbation of the lens |
| Vortex (Axion) | `vort` | Fuzzy / Axion dark matter vortex substructure |

**Primary metric:** Macro-averaged AUC (One-vs-Rest ROC)

---

## Repository Structure

```
├── Common_Test.ipynb       # Common Task  — 3 classical CNN architectures
├── Quantum_ML.ipynb        # Specific Task — 6 quantum ML experiments
└── README.md
```

---

## Notebook 1 — Common Task: Classical Transfer Learning

**`Common_Test.ipynb`** · Full dataset (30,000 train / 7,500 val images)

### Models Compared

| # | Architecture | Params | Key Design Choice |
|---|---|---|---|
| 1 | **EfficientNet-B3** | 11.48M | Compound scaling — optimal accuracy per parameter |
| 2 | **ResNet-50** | 24.55M | Skip connections — gradient highway through deep nets |
| 3 | **ConvNeXt-Tiny** | 28.21M | Transformer-inspired CNN — 7×7 depthwise conv, LayerNorm, GELU |

All three use **weight-averaging grayscale adaptation** — the pretrained 3-channel RGB stem weights are averaged across channels into a single channel, preserving ImageNet edge and texture detectors while accepting 1-channel scientific images.

### Two-Phase Transfer Learning Strategy

```
Phase 1 (5 epochs)   — Backbone FROZEN
    Optimizer : AdamW  |  LR = 3e-3  |  only classification head trains
    Purpose   : Stabilise the randomly-initialised head before touching pretrained features

Phase 2 (25 epochs)  — ALL layers UNFROZEN
    Optimizer : AdamW  |  backbone LR = 3e-4  |  head LR = 3e-3  (10× differential)
    Purpose   : Gently adapt backbone features to gravitational lensing morphology
```

**Why not unfreeze immediately?** The new head starts with random weights. Its large gradients would flow backward and corrupt the pretrained representations before the head has learned anything useful. Phase 1 prevents this catastrophic forgetting.

### Physics-Motivated Augmentations

| Augmentation | Physical Justification |
|---|---|
| Horizontal + Vertical flip | Lens plane geometry is mirror-symmetric |
| Random 90° rotation | Gravitational lensing has full rotational symmetry |
| Gaussian noise (σ=0.02) | Simulates CCD read-out noise in telescope images |
| **No** random crop | Would clip the Einstein ring, destroying morphological features |
| **No** colour jitter | Single-channel scientific images — no colour information |

### Regularisation Stack

| Technique | Setting | What It Prevents |
|---|---|---|
| Label smoothing | 0.10 | Overconfident predictions → better AUC calibration |
| Weight decay (AdamW) | 1e-4 | Large weights → overfitting / memorisation |
| Dropout | 0.4 + 0.3 | Co-adaptation of neurons |
| Gradient clipping | max_norm=1.0 | Exploding gradients → unstable Phase 2 training |
| Cosine annealing LR | eta_min=1e-6 | Sharp LR cliffs → smooth decay to near-zero |

### Results

| Model | Params | No-Sub AUC | CDM AUC | Axion AUC | **Macro AUC** | Val Acc |
|-------|--------|-----------|---------|-----------|-----------|---------|
| **EfficientNet-B3** | 11.48M | 0.9941 | 0.9901 | 0.9974 | **0.9939** | 95.96% |
| ResNet-50 | 24.55M | — | — | — | **[update after rerun]** | — |
| ConvNeXt-Tiny | 28.21M | — | — | — | **[update after rerun]** | — |

> ResNet-50 and ConvNeXt-Tiny are being retrained with a corrected DataLoader configuration (`pin_memory=False`, `num_workers=0`). The previous run had a PyTorch multiprocessing cleanup bug on Colab that collapsed Phase 2 training. Updated results are in the uploaded notebook.

**EfficientNet-B3 is the strongest model** at Macro AUC = **0.9939** — achieving near-perfect separation of all three dark matter substructure classes with only 11.48M parameters, the smallest of the three architectures.

---

## Notebook 2 — Specific Task: Quantum Machine Learning

**`Quantum_ML.ipynb`** · Subsampled dataset (1,500 train / 300 val — 500 per class)

Six quantum ML experiments, each building on the diagnosis of the previous one. The notebook traces a clear arc from a failing VQC baseline to a working hybrid architecture, with every design decision grounded in the published literature.

### Experiments

| # | Method | Key Idea |
|---|---|---|
| 1 | **Angle Encoding VQC** | 8 PCA features → 8 qubits via RY gates |
| 2 | **Amplitude Encoding VQC** | 16 features exponentially compressed into 4 qubits |
| 3 | **Trainable Quantum Kernel SVM** | Learned quantum feature map + classical convex SVM |
| 4 | **Fine-tuned CNN + VQC** | Partially unfrozen ResNet-18 → trainable projector → VQC |
| 5 | **Data Re-uploading VQC** | Features re-encoded at every circuit layer |
| 6 | **Fine-tuned CNN + Re-uploading VQC** | Exp 4 backbone + Exp 5 circuit — the strongest hybrid |

### The Core Problem: Barren Plateaus

The original Experiments 1–3 all achieved AUC ≈ 0.50 — random chance. Root cause: **barren plateaus**. Under random initialisation, gradient variance vanishes exponentially as ~2⁻ⁿ where n is the number of qubits (McClean et al. 2018). With 8 qubits and 4 layers, the gradient magnitude was ~2⁻⁸ ≈ 0.004 — too small to train. This was confirmed empirically by the gradient variance diagnostic in the notebook.

**Three fixes applied, each from published research:**

| Fix | Paper | What It Does |
|-----|-------|--------------|
| Near-zero initialisation | Grant et al. 2019 | `uniform(-0.01, 0.01)` keeps effective depth shallow at epoch 0 |
| Local cost function | Cerezo et al. 2021 | Measuring all 8 qubits (not 3) changes decay from exponential to polynomial |
| Reduced circuit depth | McClean et al. 2018 | N_LAYERS 4→2: depth-4 on 8 qubits forms an approximate 2-design |

The notebook includes a **gradient variance diagnostic** that runs before training and empirically shows the gradient magnitude difference between random and near-zero initialisation.

### Experiment 3 — Kernel Fix

The original kernel circuit computed a raw angle-space inner product with zero trainable parameters. This is not a quantum kernel in any useful sense — it gives AUC=0.50 by construction. The fix adds 2 layers of trainable `Rot` gates between the two `AngleEmbedding` calls, creating a learned feature map. The SVM then classifies in this learned quantum space rather than raw angle space. Kernel-Target Alignment (KTA) is computed after training to verify the kernel genuinely separates the classes.

### Experiment 6 — Key Architecture

```
150×150 grayscale image
    ↓  Reshape → resize 224×224 → 3-channel → ImageNet normalise
    ↓  ResNet-18 (layer1+2 frozen, layer3+4 trainable) → 512-dim features
    ↓  Trainable projector: Linear(512→64) → ReLU → Dropout(0.3) → Linear(64→8) → Tanh → [0,π]
    ↓  Re-uploading VQC: [AngleEmbedding + Rot gates + CNOT ring] × 2 layers, near-zero init
    ↓  Linear(8→3) → 3-class softmax
    [Differential LRs: 1e-4 backbone / 1e-3 projector / 5e-3 VQC head]
```

The trainable projector replaces PCA — it learns which 8 dimensions of the 512-dim CNN feature space are most discriminative for the quantum classifier, rather than selecting the 8 highest-variance PCA directions (which are not necessarily the most useful for classification).

### Results

| Method | No-Sub | CDM | Axion | **Macro AUC** | Δ vs Exp 1 |
|--------|--------|-----|-------|-----------|------------|
| Angle VQC (Exp 1) | 0.4921 | 0.5516 | 0.5336 | 0.5303 | — |
| Amplitude VQC (Exp 2) | 0.5475 | 0.5022 | 0.5418 | 0.5305 | +0.0002 |
| Quantum Kernel SVM (Exp 3) | 0.5482 | 0.4982 | 0.5285 | 0.5249 | −0.0054 |
| CNN + VQC fine-tuned (Exp 4) | 0.6471 | 0.5485 | 0.5840 | 0.5932 | +0.0629 |
| Re-uploading VQC (Exp 5) | 0.5477 | 0.5178 | 0.5182 | 0.5279 | −0.0024 |
| **CNN + Re-upload VQC (Exp 6)** | **0.6877** | **0.6449** | **0.5320** | **0.6215** | **+0.0912** |

**Best model: CNN + Re-uploading VQC (Exp 6) — Macro AUC = 0.6215 (+17.2% over Exp 1)**

> Note: All quantum experiments use `default.qubit` (classical simulator) with 1,500 training images due to exponential simulation cost. These results demonstrate approach viability, not a quantum speedup claim.

### Why These AUC Values Are Meaningful

A random classifier on a balanced 3-class problem achieves AUC = 0.50. Starting from 0.53 in Exp 1 and reaching 0.62 in Exp 6 represents a genuine learning signal — not a large one, but a real one — on a quantum simulator with only 1,500 samples and 8 qubits. The limitation is not the quantum approach itself but the simulation cost, which forces severe subsampling. On a 20+ qubit system with the full 30,000-image dataset, a properly designed QCNN (Pesah et al. 2021) or equivariant QNN (Schatzki et al. 2024) would be expected to perform substantially better.

---

## Note on Training Output Warnings (Common_Test.ipynb)

The notebook outputs may contain warnings including:

- `torch.cuda.amp.autocast` deprecation notices
- PIL image conversion messages
- `TqdmWarning` about nested progress bars

**None of these affect results.** They are cosmetic side effects of PyTorch's API migration from `torch.cuda.amp` to `torch.amp`, PIL's intermediate format handling for `.npy` arrays, and tqdm version differences in Colab. All AUC scores, ROC curves, confusion matrices, and classification reports are computed correctly. Outputs are shown as-is to reflect honest, unedited training runs.

---

## Setup & Reproduction

### Requirements

```bash
pip install pennylane pennylane-lightning torch torchvision \
            scikit-learn matplotlib seaborn tqdm gdown pillow
```

### Dataset

Both notebooks auto-download their datasets from Google Drive when run on Colab.

| Notebook | Google Drive ID |
|----------|----------------|
| `Common_Test.ipynb` | `149sa4C5jXNARESpsgm3HWXfNxuUS_kAy` |
| `Quantum_ML.ipynb` | `1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ` |

### Running

1. Open in Google Colab — Runtime → Change runtime type → **GPU (T4)**
2. Run all cells top to bottom
3. Dataset downloads automatically via `gdown`
4. Outputs saved to `outputs/` (classical) or `q_outputs/` (quantum)

### Estimated Runtimes on Colab T4

| Notebook | Runtime | Notes |
|----------|---------|-------|
| `Common_Test.ipynb` | ~45–60 min | 3 models × 30 epochs × 30,000 images |
| `Quantum_ML.ipynb` | ~60–90 min | 6 experiments × 30 epochs × 1,500 images (quantum simulator) |

---

## References

### Core quantum ML papers (directly applied in Quantum_ML.ipynb)

| Citation | Where Used |
|----------|-----------|
| McClean, J.R. et al. (2018). *Barren plateaus in quantum neural network training landscapes.* Nature Communications. [doi:10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4) | Root cause diagnosis of Exp 1–3 failure |
| Grant, E. et al. (2019). *An initialization strategy for addressing barren plateaus in parameterized quantum circuits.* Quantum. [arXiv:1903.05076](https://arxiv.org/abs/1903.05076) | Near-zero initialisation fix |
| Cerezo, M. et al. (2021). *Cost function dependent barren plateaus in shallow parametrized quantum circuits.* Nature Communications. [doi:10.1038/s41467-021-21728-w](https://doi.org/10.1038/s41467-021-21728-w) | Local cost function fix (measure all qubits) |
| Pérez-Salinas, A. et al. (2020). *Data re-uploading for a universal quantum classifier.* Quantum. [arXiv:1907.02085](https://arxiv.org/abs/1907.02085) | Basis for Experiment 5 |
| Skolik, A. et al. (2021). *Layerwise learning for quantum neural networks.* Quantum Machine Intelligence. [arXiv:2006.14904](https://arxiv.org/abs/2006.14904) | Supports shallow circuit + staged training strategy |
| Schuld, M. (2021). *Supervised quantum machine learning models are kernel methods.* Physical Review A. [arXiv:2101.11020](https://arxiv.org/abs/2101.11020) | Justifies Experiment 3 kernel redesign |
| Huang, H.Y. et al. (2021). *Power of data in quantum machine learning.* Nature Communications. [arXiv:2101.05440](https://arxiv.org/abs/2101.05440) | Quantum kernel advantage theory |
| Mari, A. et al. (2020). *Transfer learning in hybrid classical-quantum neural networks.* Quantum. [arXiv:2009.09367](https://arxiv.org/abs/2009.09367) | Basis for Experiments 4 and 6 |

### Future directions mentioned in conclusions

| Citation | Relevance |
|----------|-----------|
| Pesah, A. et al. (2021). *Absence of barren plateaus in quantum convolutional neural networks.* Physical Review X. [doi:10.1103/PhysRevX.11.041011](https://doi.org/10.1103/PhysRevX.11.041011) | QCNN — guaranteed trainable, no barren plateau |
| Schatzki, L. et al. (2024). *Theoretical guarantees for permutation-equivariant quantum neural networks.* PRX Quantum. [doi:10.1103/PRXQuantum.5.020328](https://doi.org/10.1103/PRXQuantum.5.020328) | Equivariant QNN exploiting SO(2) lensing symmetry |
| Cong, I. et al. (2019). *Quantum convolutional neural networks.* Nature Physics. [doi:10.1038/s41567-019-0648-8](https://doi.org/10.1038/s41567-019-0648-8) | QCNN avoids PCA/projector bottleneck |

### Classical deep learning (applied in Common_Test.ipynb)

| Citation | Relevance |
|----------|-----------|
| Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking model scaling for CNNs.* ICML. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946) | EfficientNet-B3 architecture |
| He, K. et al. (2016). *Deep residual learning for image recognition.* CVPR. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | ResNet-50 architecture |
| Liu, Z. et al. (2022). *A ConvNet for the 2020s.* CVPR. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545) | ConvNeXt-Tiny architecture |
| Loshchilov, I. & Hutter, F. (2019). *Decoupled weight decay regularization.* ICLR. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) | AdamW optimiser |

### DeepLense and gravitational lensing

| Citation | Relevance |
|----------|-----------|
| Mishra-Sharma, S. & Cranmer, K. (2022). *Strong lensing source reconstruction using continuous neural fields.* NeurIPS. [arXiv:2206.14820](https://arxiv.org/abs/2206.14820) | DeepLense scientific context |
| Hezaveh, Y. et al. (2017). *Fast automated analysis of strong gravitational lenses with deep learning.* Nature. [doi:10.1038/nature23463](https://doi.org/10.1038/nature23463) | Foundational work on ML for lensing |