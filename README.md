# DHA-Rec: Dual-Horizon Attentive Recommendation Network for Sequential Item Prediction

Official benchmark implementation of DHA-Rec, a dual-horizon attentive neural architecture for sequential recommendation.

## Quick Start
```bash
pip install -r requirements.txt
pip install -e .

# Download benchmark datasets
bash scripts/download_data.sh ./data

# Run full reproduction pipeline
bash scripts/reproduce_all.sh
```

## Model Architecture
DHA-Rec decouples immediate user dynamic intention from long-term preference horizons using decoupled self-attention branches and cross-horizon attentive gating.
