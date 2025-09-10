#!/bin/bash
# EduSpend Project - Virtual Environment Activation Script

echo "=== EduSpend Project Setup ==="
echo "Activating virtual environment..."

# Navigate to project directory
cd /home/yan/Documents/Git/SDS-CP030-edu-spend/submissions/team-members/yan-cotta

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "✅ Python version: $(python --version)"
echo "✅ Project directory: $(pwd)"
echo ""
echo "Available commands:"
echo "  jupyter notebook    - Start Jupyter Notebook"
echo "  jupyter lab         - Start JupyterLab"
echo "  python <script.py>  - Run Python scripts"
echo ""
echo "To deactivate, type: deactivate"
echo "=================================="
