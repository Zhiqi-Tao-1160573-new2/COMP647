# ML and AI_comp647
Machine Learning and Artificial Intelligence Course Project

# The dataset file is too large so I uploaded a compressed package. Please unzip it and use it.

## Project Overview
This is a machine learning course project that includes data analysis, data preprocessing, and machine learning model training experiments. The project focuses on analyzing automotive data to build price prediction models and explore market patterns in the Georgian car market.

## Dataset
The project uses the `primary_features.csv` dataset containing comprehensive automotive features and price information, including:
- Vehicle specifications (mileage, year, engine volume, cylinders)
- Car features (ABS, air conditioning, navigation, etc.)
- Market data (price in Georgian Lari, manufacturer, model)
- User information and listing details

## Project Structure
```
COMP647/
├── LAB1 & LAB2.py              # Data preprocessing and cleaning pipeline
├── LAB3.py                     # Exploratory Data Analysis (EDA) and visualization
├── primary_features.csv        # Main dataset file
├── primary_features.zip        # Compressed dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── venv/                      # Python virtual environment
```

##  Features

### Data Preprocessing (LAB1 & LAB2.py)
- **Duplicate Detection**: Identifies and removes duplicate records based on key features
- **Missing Value Imputation**: 
  - Median imputation for numerical features (robust to outliers)
  - Mode imputation for categorical features (preserves distribution)
  - Smart car feature filling using similar model patterns
- **Outlier Detection**: 
  - IQR method for robust outlier identification
  - Z-score method for normal distribution-based detection
- **Data Cleaning**: Removes invalid data (negative prices, unreasonable years)
- **Feature Engineering**: Creates derived features (vehicle age, price per km, brand-model combinations)

### Exploratory Data Analysis (LAB3.py)
- **Correlation Analysis**: Bar charts showing feature-price relationships
- **Pair Plot Analysis**: Comprehensive feature interaction matrix
- **Trend Analysis**: Line charts for mileage vs price relationships
- **Distribution Analysis**: 
  - KDE plots for joint distributions
  - Histograms with density estimation
  - Violin plots for distribution shapes
  - Box plots for statistical summaries
- **Categorical Analysis**: Feature cardinality and unique value counts

## Environment Setup

### Prerequisites
- Python 3.9+
- Windows 10/11 (or compatible OS)

### Installation

#### Option 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd COMP647

# Create and activate conda environment
conda create -n comp647_py311 python=3.11
conda activate comp647_py311

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using Virtual Environment
```bash
# Clone the repository
git clone <repository-url>
cd COMP647

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows CMD:
venv\Scripts\activate.bat
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Analysis
```bash
# Activate environment first
conda activate comp647_py311  # or activate your virtual environment

# Run data preprocessing
python "LAB1 & LAB2.py"

# Run exploratory data analysis
python LAB3.py
```

### Key Output Files
- `primary_features_cleaned_final.csv` - Preprocessed data ready for modeling
- `primary_features_eda_cleaned.csv` - EDA-processed data with visualizations
- Various intermediate CSV files for analysis tracking

## Key Insights

### Research Questions Explored
1. **Price Prediction Models**: Can we accurately predict car prices using vehicle features?
2. **Feature Importance**: Which car features have the strongest impact on pricing?
3. **Market Segmentation**: How do different brands and models cluster in price ranges?
4. **Outlier Analysis**: What do price outliers reveal about the Georgian car market?

### Key Findings
- Strong negative correlation between vehicle age and price (depreciation effect)
- Moderate negative correlation between mileage and price (wear and tear)
- Positive correlation with engine volume (larger engines command higher prices)
- Car features show varying correlation strengths with price
- Clear market segmentation patterns across different vehicle categories

## Visualizations
The project includes 8 comprehensive chart types:
- **Bar Charts**: Feature correlation analysis and cardinality
- **Pair Plots**: Numerical feature relationship matrix
- **Line Charts**: Mileage vs price trend analysis
- **KDE Plots**: Joint distribution density analysis
- **Histograms**: Distribution shape analysis
- **Violin Plots**: Distribution comparison across categories
- **Box Plots**: Statistical summary comparisons
- **Probability Plots**: Normality testing and outlier impact

##  Dependencies
Core libraries used in this project:
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: scipy
- **Development**: jupyter, notebook

## Methodology

### Data Preprocessing Approach
1. **Quality Assessment**: Initial data quality checks and type validation
2. **Duplicate Handling**: Smart duplicate detection using key feature combinations
3. **Missing Value Strategy**: 
   - Car features: Similar model-based imputation
   - Numerical: Median imputation (outlier-robust)
   - Categorical: Mode imputation (distribution-preserving)
4. **Outlier Treatment**: IQR and Z-score methods with statistical justification
5. **Feature Engineering**: Domain-specific derived features

### EDA Methodology
1. **Correlation Analysis**: Pearson correlation for linear relationships
2. **Distribution Analysis**: Multiple visualization techniques for comprehensive understanding
3. **Trend Analysis**: Time-series-like analysis for continuous variables
4. **Categorical Analysis**: Cardinality and frequency analysis
5. **Statistical Validation**: Probability plots and normality testing

## Technical Highlights
- **Robust Preprocessing**: Handles real-world data quality issues
- **Comprehensive EDA**: 8 different visualization types for thorough analysis
- **Statistical Rigor**: Multiple outlier detection methods with justification
- **Feature Engineering**: Domain knowledge integration for better modeling
- **Reproducible Analysis**: Fixed random seeds and clear documentation

## Assignment Requirements Coverage
 **Data Preprocessing**: Complete cleaning pipeline with detailed explanations  
 **EDA**: Comprehensive exploratory analysis with correlation studies  
 **Feature Insights**: Detailed explanations for all methodological choices  
 **Research Questions**: Clear problem formulation backed by EDA findings  

