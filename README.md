# ReasonBreak: Disrupting Hierarchical Reasoning

### Adversarial Protection for Geographic Privacy in Multimodal Reasoning Models

**[Project Page](https://jiamingzz94.github.io/reasonbreak/)** | **[Paper](https://arxiv.org/abs/2512.08503)** | **[Dataset](https://huggingface.co/datasets/jiamingzz/geo_6k)** | **[Model](https://huggingface.co/jiamingzz/reason_break)**

Official implementation of **ReasonBreak**, an adversarial framework designed to protect geographic privacy against Multimodal Large Reasoning Models (MLRMs).


## 🛠️ Installation

```bash
conda create -n reasonbreak python=3.10
conda activate reasonbreak
pip install -r requirements.txt
````

## 📂 Data Preparation

Please organize the directory as follows:

```text
ReasonBreak/
├── data/
│   ├── embedding_bank.pth        # Download from Model repo
│   ├── json/
│   │   ├── location_analysis_fixed.jsonl  # For training
│   │   └── cot_full.json                  # For inference
│   └── images/
│       ├── geo_6k/               # [Optional] Training Set
│       └── dox/                  # [Required] Evaluation Set
├── checkpoints/
│   └── weight.pth                # ReasonBreak Weights
└── ...
```

### Download Links

  * **Datasets:**
      * **[GeoPrivacy-6K (Training)](https://huggingface.co/datasets/jiamingzz/geo_6k)**: *Optional*. Download only if you intend to train the model from scratch.
      * **[DoxBench (Evaluation)](https://huggingface.co/datasets/MomoUchi/DoxBench)**: *Required*. Necessary for running the evaluation pipeline.
  * **Model Weights:**
      * **[ReasonBreak](https://huggingface.co/jiamingzz/reason_break)**: Download `weight.pth` and place it in `checkpoints/`.

### API Configuration

Create a `.env` file in `code/experiment/` for evaluation:

```bash
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

## 🚀 Quick Start

Run the unified script `run.sh`. **Choose ONE of the following options:**

### Option 1: Full Pipeline (Recommended)

Generate adversarial samples and immediately evaluate performance.

```bash
bash run.sh --mode adv --step all --dox_path /path/to/data/images/dox
```

### Option 2: Baseline Evaluation

Evaluate MLRMs on clean (unprotected) images.

```bash
bash run.sh --mode clean --step eval --dox_path /path/to/data/images/dox
```

### Option 3: Generation Only

Generate adversarial images without evaluation.

```bash
bash run.sh --mode adv --step gen --dox_path /path/to/data/images/dox
```

## 🏋️ Training (Optional)

To train from scratch using **GeoPrivacy-6K**:

1.  Ensure `geo_6k` images are in `data/images/geo_6k`.
2.  Run the training script (initialization weights will be handled automatically):

<!-- end list -->

```bash
bash scripts/train.sh
```

## 📜 Citation

```bibtex
@article{zhang2025reasonbreak,
  title={Disrupting Hierarchical Reasoning: Adversarial Protection for Geographic Privacy in Multimodal Reasoning Models},
  author={Zhang, Jiaming and Wang, Che and Cao, Yang and Huang, Longtao and Lim, Wei Yang Bryan},
  journal={arXiv preprint arXiv:2512.08503},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License.

````

