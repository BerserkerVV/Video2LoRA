# Video2LoRA

### Unified Semantic-Controlled Video Generation via Per-Reference-Video LoRA

**CVPR 2026 Findings**

Official implementation of **Video2LoRA**, a unified framework for semantic-controlled video generation via **per-reference-video LoRA predicted by a HyperNetwork**.

> Video2LoRA enables semantic video generation by dynamically predicting lightweight LoRA adapters from reference videos, without requiring per-condition fine-tuning.

---

# 🔥 Highlights

Video2LoRA introduces a new paradigm for **semantic-controlled video generation**.

Instead of training separate models or LoRA adapters for each semantic condition (e.g., visual effects, camera motion, style), our framework **predicts semantic-specific LoRA weights directly from a reference video**.

Key features:

* 🎬 **Reference-driven semantic video generation**
* ⚡ **Ultra-lightweight LoRA (<50 KB per semantic condition)**
* 🧠 **Transformer-based HyperNetwork for LoRA prediction**
* 🌍 **Strong zero-shot generalization**
* 🧩 **Unified framework across heterogeneous semantic controls**

---

# 🧠 Method Overview

![framework](hyper_crop.pdf)

Video2LoRA consists of three key components:

## 1. LightLoRA Representation

We introduce **LightLoRA**, a compact LoRA formulation that decomposes the standard LoRA matrices:

[
A = A_{\text{aux}} A_{\text{pred}}, \quad
B = B_{\text{pred}} B_{\text{aux}}
]

Where:

* (A_{\text{aux}}, B_{\text{aux}}): trainable auxiliary matrices
* (A_{\text{pred}}, B_{\text{pred}}): predicted by the HyperNetwork

This design significantly reduces parameter size while preserving semantic adaptability.

Each semantic condition requires **less than 50 KB** parameters.

---

## 2. HyperNetwork for LoRA Prediction

A **Transformer-based HyperNetwork** predicts semantic-specific LoRA weights conditioned on a reference video.

Pipeline:

```
Reference Video
      ↓
3D VAE Encoder
      ↓
Spatio-temporal features
      ↓
Transformer Decoder
      ↓
Predicted LoRA weights
```

These predicted LoRA modules are injected into the frozen diffusion backbone.

---

## 3. End-to-End Diffusion Training

Unlike prior methods that require:

* pretrained semantic LoRA weights
* multi-stage training pipelines

Video2LoRA is trained **end-to-end using only the standard diffusion objective**.

---

# 🎬 Results

We evaluate Video2LoRA on the **Open-VFX dataset**.

| Metric              | Ours      | Best Baseline |
| ------------------- | --------- | ------------- |
| FVD ↓               | **1568**  | 1679          |
| Dynamic Degree ↑    | **0.78**  | 0.71          |
| Motion Smoothness ↑ | **98.50** | 98.24         |
| Aesthetic Quality ↑ | **0.565** | 0.537         |

Our method consistently improves:

* visual fidelity
* motion coherence
* semantic accuracy

---

# 🌍 Zero-Shot Semantic Generation

Video2LoRA generalizes well to **unseen semantic conditions**.

Even when encountering **out-of-domain visual effects**, the model can generate semantically aligned videos based on reference videos.

Example semantic controls include:

* visual effects (VFX)
* camera motion
* object stylization
* character transformations
* artistic styles

---

# 📂 Dataset

Our training dataset is collected from multiple sources:

* **Open-VFX**
* Higgsfield
* PixVerse
* public online video resources

Dataset statistics:

* ~4K video samples
* 200+ semantic categories
* multiple effect types and styles

Dataset structure:

```
datasets/
    videos/
    annotations/
```

---

# ⚙️ Installation

## Clone repository

```bash
git clone https://github.com/BerserkerVV/Video2LoRA_new.git
cd Video2LoRA_new
```

## Create environment

```bash
conda create -n video2lora python=3.10
conda activate video2lora
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

Train Video2LoRA:

```bash
bash scripts/train.sh
```

or

```bash
python train.py --config configs/train.yaml
```

Training setup:

| Item       | Value            |
| ---------- | ---------------- |
| Backbone   | CogVideoX-I2V-5B |
| GPUs       | 8 × NVIDIA A800  |
| Iterations | 20K              |
| Frames     | 49               |
| FPS        | 8                |
| Resolution | 480×720          |

Only the following parameters are trained:

* HyperNetwork
* auxiliary matrices

The diffusion backbone remains **frozen**.

---

# 🎥 Inference

Generate a video using a reference video:

```bash
python inference.py \
    --reference path/reference.mp4 \
    --prompt "a person dissolving into particles"
```

---

# 📊 Ablation Study

We analyze the impact of:

* LightLoRA dimensionality `(a,b)`
* iterative refinement steps `k`

Best configuration:

```
a = 100
b = 50
k = 4
```

---

# 📖 Citation

If you find our work useful, please cite:

```
@inproceedings{video2lora2026,
  title={Video2LoRA: Unified Semantic-Controlled Video Generation via Per-Reference-Video LoRA},
  author={...},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

# 🙏 Acknowledgements

This project builds upon several excellent open-source works:

* CogVideoX
* Diffusion Transformers
* LoRA

We thank the authors for their contributions to the community.

---

# 📬 Contact

If you have any questions, feel free to open an issue or contact the authors.

---

## ⭐ Star the Repo

If you find this project useful, please consider starring the repository to support our work.
