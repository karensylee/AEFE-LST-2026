#!/bin/bash
# Train BLAM-ALL variants and run SHAP analysis

echo "==========================================="
echo "STARTING BLAM-ALL TRAINING AND ANALYSIS"
echo "==========================================="

# 1. Train BLAM-ALL (Default - All Data)
echo "Training BLAM-ALL (Default)..."
python3 main.py --model_type BLAM-ALL

# 2. Train BLAM-ALL-CLEAR (Clear Sky Only)
echo "Training BLAM-ALL-CLEAR..."
python3 main.py --model_type BLAM-ALL-CLEAR

# 3. Train BLAM-ALL-CLOUDY (Cloudy Sky Only)
echo "Training BLAM-ALL-CLOUDY..."
python3 main.py --model_type BLAM-ALL-CLOUDY

# 4. Run SHAP Analysis
echo "Running TreeSHAP Analysis..."
python3 analysis/blam_shap_beeswarm.py

echo "==========================================="
echo "ALL TASKS COMPLETED"
echo "==========================================="
