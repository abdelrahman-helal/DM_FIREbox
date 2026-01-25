# Inferring Dark Matter Halo Mass in the FIREbox Simulation with Graph Neural Networks

## 🎓 Capstone Project

This repository contains the code and analysis for my capstone project on **"Inferring Dark Matter Halo Mass in the FIREbox Simulation with Graph Neural Networks"**.

## 📖 About the Project

This project explores the application of Graph Neural Networks (GNNs) to predict dark matter halo masses using data from the FIREbox cosmological simulation. By treating halos as nodes in a graph and connecting them based on spatial proximity, we can leverage the relational structure of the cosmic web to improve mass predictions.

## 🐳 Docker: Train with Volume Mounts

Build the image:

```
docker build -t dm-firebox .
```

Run training with local data and outputs mounted into the container:

```
docker run --rm \
  -v /path/to/local/data:/data \
  -v /path/to/local/outputs:/outputs \
  -e WANDB_API_KEY=your_key_here \
  dm-firebox \
  python scripts/train.py --config /app/configs/base.yaml
```

Notes:
- Data must be under `/path/to/local/data`, matching `configs/base.yaml` (e.g., `firebox_data/FIREbox_z=0.txt`).
- Best model is saved to `/path/to/local/outputs/best_model.pt`.
- If you leave `wandb.enabled: false`, W&B is skipped and `WANDB_API_KEY` is optional.
- Optional W&B env vars: `WANDB_PROJECT`, `WANDB_ENTITY`.

## 🔬 FIREbox Simulation

The FIREbox simulation is a state-of-the-art cosmological simulation developed by the **FIRE (Feedback in Realistic Environments) collaboration**. It simulates galaxies at high dynamic range in a cosmological volume, providing detailed insights into galaxy formation and evolution.

### Citation
> Robert Feldmann, Eliot Quataert, Claude-André Faucher-Giguère, Philip F Hopkins, Onur Çatmabacak, Dušan Kereš, Luigi Bassini, Mauro Bernardini, James S Bullock, Elia Cenci, Jindra Gensior, Lichen Liang, Jorge Moreno, Andrew Wetzel, **FIREbox: simulating galaxies at high dynamic range in a cosmological volume**, *Monthly Notices of the Royal Astronomical Society*, Volume 522, Issue 3, July 2023, Pages 3831–3860, https://doi.org/10.1093/mnras/stad1205

