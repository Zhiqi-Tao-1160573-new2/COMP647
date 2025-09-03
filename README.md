# ML and AI_comp647
Machine Learning and Artificial Intelligence Course Project

## Project Overview
This is a machine learning course project that includes data analysis, data preprocessing, and machine learning model training experiments.

## Project Structure
```
COMP647/
├── LAB1 & LAB2.py      # Lab 1 & 2: Data preprocessing and analysis
├── LAB3.py             # Lab 3: Machine learning models
├── pandas_report.py    # Pandas data analysis report
├── primary_features.zip # Dataset file
├── requirements.txt    # Complete dependency list
├── requirements_simple.txt # Simplified dependency list
├── LAB1 & LAB2_中文.py # Chinese version of Lab 1 & 2
├── LAB3_中文.py        # Chinese version of Lab 3

└── venv/              # Python virtual environment
```

## Environment Requirements
- Python 3.9+
- Windows 10/11

## Quick Start

### Using conda environment (Recommended)
1. Open Command Prompt or PowerShell
2. Navigate to project directory: `cd D:\Programs\github\COMP647`
3. Activate conda environment: `conda activate comp647_py311`
4. Run Python scripts

### Using virtual environment (Alternative)
1. Open Command Prompt or PowerShell
2. Navigate to project directory: `cd D:\Programs\github\COMP647`
3. Activate virtual environment:
   - CMD: `venv\Scripts\activate.bat`
   - PowerShell: `.\venv\Scripts\Activate.ps1`
4. Run Python scripts

## Available Commands
```bash
# Run experiment scripts
python "LAB1 & LAB2.py"  # Note the spaces in filename
python LAB3.py
python pandas_report.py

# Start Jupyter environment
jupyter notebook
jupyter lab
```

## Installed Core Libraries
- **Data Processing**: pandas, numpy
- **Data Visualization**: matplotlib, seaborn
- **Scientific Computing**: scipy
- **Development Environment**: jupyter, notebook

## Dataset
The project uses the `primary_features.csv` dataset, which contains car features and price information.

## Important Notes
1. Ensure environment is activated before first run
2. Recommended to use conda environment `comp647_py311` (supports ydata-profiling)
3. Use quotes around filenames that contain spaces

## Troubleshooting
- If pip installation fails, try updating pip: `python -m pip install --upgrade pip`
- If some library versions are incompatible, use versions from `requirements_simple.txt`
