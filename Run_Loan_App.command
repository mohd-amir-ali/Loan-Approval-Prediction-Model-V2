#!/bin/bash
cd "$(dirname "$0")"
echo "=================================================================="
echo "  CRED-TREE // CART Decision Tree Loan Underwriting Engine"
echo "  Opening Interactive Platform in Default Browser..."
echo "=================================================================="
open "index.html"
python3 app.py
