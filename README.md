# Alternative-NDT

Hybrid classification and regression method by combining Neural Network and Decision Tree.

## Setup

1. **Get Cursor (recommended)**  
   Download the editor: [Cursor](https://cursor.com/product).

2. **Install dependencies**  
   From the project root:
   ```bash
   pip install -r requirements.txt
   ```

## Run an example

**Entry point:** `tf_MINIST.py` is the main example to run:

```bash
python tf_MINIST.py
```

Other examples: `tf_Reuters`, `tf_SVHN`, `tf_CIFAR10`, `tf_Comparison`, `tf_circle`.

## How it works

- **`framework/ndtFunc.py`** — Builds the weight matrices from a fitted decision tree (splits, thresholds, children, leaf values). These weights encode the tree so it can be expressed as differentiable layers.

- **`framework/tfNDT.py`** — Builds the Neural Decision Tree (NDT): loads data, fits a decision tree, uses `ndtFunc` to get weights, then trains those weights with TensorFlow so the tree is fine-tuned like a small neural network.

Flow: **Decision tree → ndtFunc (weights) → tfNDT (NDT model + training)**.
