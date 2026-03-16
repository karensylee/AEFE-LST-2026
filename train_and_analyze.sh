#!/bin/bash
# Train SXE-ALL variants and run SHAP analysis

echo "==========================================="
echo "STARTING SXE-ALL TRAINING AND ANALYSIS"
echo "==========================================="

# 1. Train SXE-ALL (Default - All Data)
echo "Training SXE-ALL (Default)..."
python3 main.py --model_type SXE-ALL

# 2. Train SXE-ALL-CLEAR (Clear Sky Only)
echo "Training SXE-ALL-CLEAR..."
python3 main.py --model_type SXE-ALL-CLEAR

# 3. Train SXE-ALL-CLOUDY (Cloudy Sky Only)
echo "Training SXE-ALL-CLOUDY..."
python3 main.py --model_type SXE-ALL-CLOUDY

# 4. Run SHAP Analysis
echo "Running TreeSHAP Analysis..."
python3 analysis/blam_shap_beeswarm.py

echo "==========================================="
echo "ALL TASKS COMPLETED"
echo "==========================================="
