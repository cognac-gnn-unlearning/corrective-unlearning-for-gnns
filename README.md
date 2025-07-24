<h1 align="center">
 A <i>Cognac</i> Shot To Forget Bad Memories: Corrective Unlearning for GNNs [ICML 2025]
</h1>

<p align="center">
 <strong>🎉 Accepted to ICML 2025 🎉</strong>
</p>

<p align="center">
 <a href="https://arxiv.org/abs/2412.00789">📄 Paper</a> •
 <a href="#citation">📝 Citation</a>
</p>

### Authors
[Varshita Kolipaka](https://github.com/varshitakolipaka)<sup>*1</sup>, [Akshit Sinha](https://github.com/viciousAegis)<sup>*1</sup>, [Debangan Mishra](https://github.com/Debangan-MishraIIIT)<sup>1</sup>, [Sumit Kumar](https://github.com/Sumitkk10)<sup>1</sup>, [Arvindh Arun](https://github.com/arvindh75)<sup>1,2</sup>, [Shashwat Goel](https://github.com/shash42)<sup>†1,3,4</sup>, [Ponnurangam Kumaraguru](https://precog.iiit.ac.in/)<sup>1</sup>

<sup>1</sup>IIIT Hyderabad • <sup>2</sup>Institute for AI, University of Stuttgart • <sup>3</sup>Max Planck Institute for Intelligent Systems • <sup>4</sup>ELLIS Institute Tübingen

<sup>*</sup>Equal contribution. <sup>†</sup>Equal advising.

---

## Quick Start

### Installation (using uv - recommended)

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup environment**:
   ```bash
   git clone https://github.com/your-repo/corrective-unlearning-for-gnns
   cd corrective-unlearning-for-gnns
   uv sync
   ```

### Alternative Installation (using conda/pip)

1. **Create environment**:
   ```bash
   conda create --name cognac_env python=3.12
   conda activate cognac_env
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Understanding the Evaluation Metrics

Our evaluation follows the paper's methodology with two key metrics (please refer to the paper for detailed definitions):

### Forget Accuracy (Acc_aff)
- **Definition**: Accuracy on the affected distribution (data that should be "forgotten")
- **Goal**: Unlearning should **increase** this score
- **Interpretation**: **Higher is better** - indicates better unlearning performance

### Utility Accuracy (Acc_rem) 
- **Definition**: Accuracy on the remaining distribution (clean/unaffected data)
- **Goal**: Should remain **unaffected** by unlearning
- **Interpretation**: **Higher is better** - indicates preserved model utility

### Key Takeaway
- **Forget**: Higher = Better unlearning
- **Utility**: Higher = Better retention of original performance
- The ideal unlearning method maximizes both metrics

## Reproducing Results

### Recommended: Automated Best Variant Selection
**Use this script to automatically run both Cognac variants and get the best result:**

```bash
python run_cognac_best.py --dataset Cora --gnn gcn --df_size 0.3 --attack_type label
```

This script will:
1. Run hyperparameter tuning for both `cognac` and `cognac-descent`
2. Run main experiments for both variants
3. Compare results and report the best performing variant
4. Provide a clear recommendation for your specific configuration

### Manual Method (if you prefer step-by-step control)

#### Step 1: Hyperparameter Tuning (Required)
**You must run this first** to find optimal hyperparameters:

```bash
python hp_tune.py --dataset Cora --gnn gcn --attack_type label --unlearning_model cognac
```

#### Step 2: Run Experiments
After hyperparameter tuning, run the main experiments:

```bash
python main.py --dataset Cora --gnn gcn --df_size 0.3 --attack_type label --unlearning_model cognac
```

### Our Method: Cognac
We propose **Cognac**, which comes in two variants:
- `cognac`: Full method with ascent and descent
- `cognac-descent`: Descent-only variant

**The `run_cognac_best.py` script automatically tests both and recommends the best one for your configuration.**

Manual commands for individual variants:
```bash
# Run Cognac (full method)
python main.py --dataset Cora --gnn gcn --df_size 0.3 --attack_type label --unlearning_model cognac

# Run Cognac-Descent (descent only)
python main.py --dataset Cora --gnn gcn --df_size 0.3 --attack_type label --unlearning_model cognac-descent
``` 

## Available Methods

### Our Methods
- **`cognac`**: Full Cognac method with ascent and descent phases
- **`cognac-descent`**: Cognac descent-only variant

### Baseline Methods
- `retrain`: Complete retraining from scratch
- `gnndelete`: GNNDelete unlearning method
- `gradient_ascent`: Simple gradient ascent approach
- `scrub`: SCRUB unlearning method
- `megu`: MEGU unlearning approach
- `ssd`: Selective Synaptic Dampening
- `gif`: GIF 
- `utu`: Unlink to Unlearn
- `acdc`: Ascent-Descent

## Key Arguments

```bash
python main.py [OPTIONS]
```

### Essential Arguments
- `--dataset`: Dataset to use (e.g., `Cora`, `CiteSeer`, `PubMed`)
- `--gnn`: GNN architecture (`gcn`, `gat`, `gin`)
- `--attack_type`: Attack method (`label`, `edge`, `trigger`)
- `--unlearning_model`: Unlearning method (see available methods above)
- `--df_size`: Forgetting fraction (e.g., `0.3` for 30%)

### Common Usage Examples
```bash
# Recommended: Automated best variant selection
python run_cognac_best.py --dataset Cora --gnn gcn --df_size 0.3 --attack_type label

# Manual: Basic Cognac run
python main.py --dataset Cora --gnn gcn --df_size 0.3 --attack_type label --unlearning_model cognac

# Manual: Cognac-descent variant
python main.py --dataset Cora --gnn gcn --df_size 0.3 --attack_type label --unlearning_model cognac-descent

# Baseline comparison
python main.py --dataset Cora --gnn gcn --df_size 0.3 --attack_type label --unlearning_model retrain
```

For complete argument list, run: `python main.py --help`

## Comprehensive Baselining Guide

### Using the Automated Script (Recommended)

The `run_cognac_best.py` script automatically finds the best Cognac variant for your configuration:

```bash
# Basic usage - minimal arguments
python run_cognac_best.py --dataset Cora --gnn gcn --attack_type label --df_size 0.3

# Full configuration for research paper reproduction
python run_cognac_best.py \
    --dataset Cora \
    --gnn gcn \
    --attack_type label \
    --df_size 0.3 \
    --random_seed 42 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --hidden_dim 64 \
    --training_epochs 1208 \
    --unlearning_epochs 200

# Different datasets and attack types
python run_cognac_best.py --dataset CiteSeer --gnn gat --attack_type edge --df_size 0.5
python run_cognac_best.py --dataset PubMed --gnn gin --attack_type trigger --df_size 0.2

# Advanced control options
python run_cognac_best.py --dataset Cora --gnn gcn --attack_type label --df_size 0.3 --skip-hp-tune  # Skip HP tuning
python run_cognac_best.py --dataset Cora --gnn gcn --attack_type label --df_size 0.3 --only-compare   # Only compare existing results
```

### Manual Baselining (Step-by-Step)

For researchers who want full control over each step:

#### 1. Hyperparameter Tuning for Both Variants

```bash
# Tune Cognac (full method)
python hp_tune.py \
    --dataset Cora \
    --gnn gcn \
    --attack_type label \
    --unlearning_model cognac \
    --df_size 0.3 \
    --random_seed 42

# Tune Cognac-Descent (descent only)
python hp_tune.py \
    --dataset Cora \
    --gnn gcn \
    --attack_type label \
    --unlearning_model cognac-descent \
    --df_size 0.3 \
    --random_seed 42
```

#### 2. Run Main Experiments for Both Variants

```bash
# Run Cognac (full method)
python main.py \
    --dataset Cora \
    --gnn gcn \
    --attack_type label \
    --unlearning_model cognac \
    --df_size 0.3 \
    --random_seed 42 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --hidden_dim 64 \
    --training_epochs 1208 \
    --unlearning_epochs 200

# Run Cognac-Descent (descent only)
python main.py \
    --dataset Cora \
    --gnn gcn \
    --attack_type label \
    --unlearning_model cognac-descent \
    --df_size 0.3 \
    --random_seed 42 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --hidden_dim 64 \
    --training_epochs 1208 \
    --unlearning_epochs 200
```

#### 3. Compare with Baseline Methods

```bash
# Retrain baseline
python hp_tune.py --dataset Cora --gnn gcn --attack_type label --unlearning_model retrain --df_size 0.3
python main.py --dataset Cora --gnn gcn --attack_type label --unlearning_model retrain --df_size 0.3

# GNNDelete baseline
python hp_tune.py --dataset Cora --gnn gcn --attack_type label --unlearning_model gnndelete --df_size 0.3
python main.py --dataset Cora --gnn gcn --attack_type label --unlearning_model gnndelete --df_size 0.3

# SCRUB baseline
python hp_tune.py --dataset Cora --gnn gcn --attack_type label --unlearning_model scrub --df_size 0.3
python main.py --dataset Cora --gnn gcn --attack_type label --unlearning_model scrub --df_size 0.3

# MEGU baseline
python hp_tune.py --dataset Cora --gnn gcn --attack_type label --unlearning_model megu --df_size 0.3
python main.py --dataset Cora --gnn gcn --attack_type label --unlearning_model megu --df_size 0.3
```

### Complete Experimental Pipeline

For comprehensive evaluation across multiple settings:

```bash
# Multiple datasets
for dataset in Cora CiteSeer PubMed; do
    python run_cognac_best.py --dataset $dataset --gnn gcn --attack_type label --df_size 0.3
done

# Multiple attack types
for attack in label edge trigger; do
    python run_cognac_best.py --dataset Cora --gnn gcn --attack_type $attack --df_size 0.3
done

# Multiple forgetting fractions
for df_size in 0.1 0.3 0.5; do
    python run_cognac_best.py --dataset Cora --gnn gcn --attack_type label --df_size $df_size
done

# Multiple GNN architectures
for gnn in gcn gat gin; do
    python run_cognac_best.py --dataset Cora --gnn $gnn --attack_type label --df_size 0.3
done

# Multiple random seeds for statistical significance
for seed in 42 123 456 789 999; do
    python run_cognac_best.py --dataset Cora --gnn gcn --attack_type label --df_size 0.3 --random_seed $seed
done
```

### Expected Output Structure

Results will be saved in the following structure:
```
logs/default/
├── Cora/
│   ├── run_logs_label_0.3_42.json      # Main results
│   ├── run_logs_label_0.3_123.json     # Different seed
│   └── ...
├── CiteSeer/
│   └── ...
└── PubMed/
    └── ...

hp_tuning/
└── hp_tuning.db                        # Hyperparameter optimization results

best_params.json                        # Best hyperparameters for each configuration
```

### Key Metrics to Compare

When baselining, focus on these metrics from the log files:
- **Test Accuracy**: Overall model performance
- **Forget Accuracy**: Performance on nodes that should be forgotten
- **Utility Accuracy**: Performance on remaining nodes
- **F1 Scores**: For imbalanced datasets
- **Training Time**: Computational efficiency

## Project Structure

- `attacks/`: Graph attack implementations
- `framework/`: Core utilities and training arguments
- `models/`: GNN architectures (GCN, GAT, GIN)
- `trainers/`: Unlearning method implementations
- `hp_tune.py`: **Required** hyperparameter optimization
- `main.py`: Main experiment runner
- `run_cognac_best.py`: **Recommended** automated script to find best Cognac variant
- `pyproject.toml`: uv project configuration
- `requirements.txt`: Alternative dependency list

## Citation

```bibtex
@misc{kolipaka2024cognacshotforgetbad,
     title={A Cognac shot to forget bad memories: Corrective Unlearning in GNNs}, 
     author={Varshita Kolipaka and Akshit Sinha and Debangan Mishra and Sumit Kumar and Arvindh Arun and Shashwat Goel and Ponnurangam Kumaraguru},
     year={2024},
     eprint={2412.00789},
     archivePrefix={arXiv},
     primaryClass={cs.LG},
     url={https://arxiv.org/abs/2412.00789}, 
}
```

## Notes

- **Hyperparameter tuning is mandatory** before running experiments
- **Use `run_cognac_best.py` for automatic best variant selection** (recommended)
- For manual runs, test both `cognac` and `cognac-descent` variants for fair comparison
- Results are stored in the `logs/` directory
- The project uses uv for modern Python dependency management
- GPU is recommended for faster training
