# Clinic no-show prediction

This project predicts whether a clinic appointment will be a no-show or a checked-in visit. `analysis.py` loads `clinic_appointments.csv`, trains one decision tree on a stratified 70/30 split, compares it with an always-checked-in baseline, applies the tree to two upcoming appointments, estimates the financial effect of overbooking, and saves the tree diagram as `decision_tree.png`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python analysis.py
```
