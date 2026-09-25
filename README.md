# Q-ROUTE AI

**On-Device Intelligent Traffic Prediction and Quantum-Inspired Route Optimization**

## Problem

Traffic changes over time, so a static shortest-path route may not reflect predicted congestion or an incident.

## Solution

Q-ROUTE AI combines traffic prediction, congestion estimation, quantum-inspired particle swarm optimization (QPSO), multi-vehicle routing, and incident-driven re-routing in an offline demonstration pipeline.

**AI predicts. QPSO optimizes.** QPSO is quantum-inspired and runs on classical hardware; this project does not use quantum hardware.

## Architecture

Traffic Data → AI Prediction → Congestion → Road Network → QPSO → Multi-Vehicle Optimization → Dynamic Re-Routing

The demo road graph is a small synthetic offline network. The UCI Metro Interstate Traffic Volume dataset represents measurements from the westbound I-94 highway segment; it does not represent an entire city.

## Technology

Python, pandas, scikit-learn, joblib, Streamlit, NetworkX, Matplotlib, and a discrete quantum-inspired QPSO search.

## Setup and commands

Use the project virtual environment on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe ai\inference.py
.\.venv\Scripts\python.exe app\demo.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe benchmarks\compare_methods.py
.\.venv\Scripts\python.exe benchmarks\ai_benchmark.py
.\.venv\Scripts\streamlit.exe run app\dashboard.py
```

To retrain the existing Random Forest baseline, run `.venv\Scripts\python.exe ai\train_model.py` from the repository root. Training overwrites `models/traffic_model.pkl`.

## Results

Benchmark scripts write measured route runs to `benchmarks/results.csv` and print actual local inference latency. No performance claims are prefilled here. The route benchmark compares a travel-time shortest-path baseline with QPSO and AI + QPSO under the script's fixed demonstration inputs.

## Snapdragon deployment

**Implemented:** Random Forest inference using scikit-learn/joblib on the current development PC, with congestion passed into route costs.

**Target / future:** Snapdragon-powered HP PC inference through Qualcomm AI Hub optimization/conversion and a compatible Snapdragon runtime/NPU. The current Random Forest pickle is not itself an AI Hub deployable artifact. A practical path is to train an additional supported lightweight neural model, export to ONNX, compare its predictions with the Random Forest, then convert/compile using the currently supported AI Hub toolchain and validate on hardware. Keep the RF baseline until parity is measured. No AI Hub conversion, Snapdragon run, or NPU benchmark is claimed.

Snapdragon/NPU execution is the target deployment configuration; current development and validation were performed on the available development PC.

## Project structure

- `ai/` inference and training scripts
- `app/` demo and Streamlit dashboard
- `benchmarks/` route and inference benchmarks
- `data/` I-94 traffic data and prepared splits
- `docs/` architecture, demo script, and deployment notes
- `models/` trained traffic model
- `qpso/` quantum-inspired optimizer
- `routing/` offline network, costs, multi-vehicle planning, rerouting
- `tests/` prototype tests
