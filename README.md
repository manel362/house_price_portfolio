#  House Price Prediction - Complete Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

**A comprehensive end-to-end regression analysis project demonstrating data science best practices: EDA, feature engineering, model training, and evaluation.**

---

##  Project Overview

This project implements a **complete machine learning pipeline** for predicting house prices using real estate data. It showcases:

 **Exploratory Data Analysis (EDA)** - Understanding data patterns and distributions  
 **Feature Engineering** - Creating meaningful predictive features  
 **Model Training** - Comparing 4 different regression algorithms  
 **Model Evaluation** - Comprehensive performance metrics and diagnostics  
 **Data Visualization** - Clear, publication-quality plots  

### Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | Linear Regression |
| **R² Score** | 0.9322 (93.2% variance explained) |
| **RMSE** | $49,185 |
| **MAE** | $39,161 |
| **Dataset Size** | 1,460 properties |

---

## 📁 Project Structure

```
house_price_portfolio/
├── house_price_analysis.py          # Main analysis script
├── README.md                         # This file
├── 01_eda_visualizations.png        # EDA plots (6 subplots)
├── 02_model_diagnostics.png         # Model evaluation plots
└── requirements.txt                 # Python dependencies
```

---

## 🔍 Key Findings

### Price Drivers
1. **Living Area** - Strongest predictor (r=0.942)
   - Each additional sq ft adds ~$120 to price
   
2. **Property Age** - Newer homes are more expensive
   - ~$500 decrease per year of age
   
3. **Garage Area** - Secondary driver (r=0.090)
   - Additional garage space increases value
   
4. **Bedrooms** - Moderate impact (r=0.193)
   - More bedrooms = higher average price

### Market Insights
- **Average Price**: $414,849
- **Price Range**: $50,000 - $878,372
- **Typical Properties**: 2,983 sq ft living area, built ~1940

---

## 📈 Model Performance Comparison

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **Linear Regression** | **0.9322** | **$49,185** | **$39,161** |
| Ridge Regression | 0.9305 | $49,807 | $39,916 |
| Random Forest | 0.9149 | $55,090 | $44,668 |
| Gradient Boosting | 0.9158 | $54,811 | $44,283 |

**Winner**: Linear Regression - Best generalization with lowest error metrics.

---

##  Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/house_price_portfolio.git
   cd house_price_portfolio
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**
   ```bash
   python house_price_analysis.py
   ```

4. **View outputs**
   - Check console output for summary statistics
   - Open `01_eda_visualizations.png` for EDA charts
   - Open `02_model_diagnostics.png` for model evaluation

---

##  Detailed Analysis

### Part 1: Exploratory Data Analysis
The analysis begins with comprehensive EDA:
- **Statistical Summary** - Mean, std dev, min/max for all features
- **Missing Values** - Detected 0 missing values (clean dataset)
- **Correlation Analysis** - Identified top features correlated with price
- **Distribution Analysis** - Examined price distribution and outliers

**Visualizations created:**
- Distribution of house prices (histogram)
- Price vs living area (scatter with trend line)
- Price vs year built (temporal analysis)
- Price by bedroom count (boxplot)
- Feature correlation heatmap
- Top features correlation chart

### Part 2: Feature Engineering
Created derived features to improve model performance:
- **Age** = 2020 - YearBuilt (captures property depreciation)
- **RemodAge** = Years since last remodel
- **TotalBaths** = Bathroom count
- **TotalSqft** = Living area + Garage area

### Part 3: Model Training
Trained 4 different regression models:
1. **Linear Regression** - Baseline model (interpretable)
2. **Ridge Regression** - L2 regularization variant
3. **Random Forest** - Tree-based ensemble
4. **Gradient Boosting** - Advanced ensemble method

**Data Split**: 80% training (1,168), 20% testing (292)

### Part 4: Model Evaluation
Comprehensive evaluation on test set:
- **R² Score** - Variance explained by model
- **RMSE** - Root Mean Squared Error (penalizes large errors)
- **MAE** - Mean Absolute Error (interpretable in dollars)
- **Residual Analysis** - Error distribution and patterns

### Part 5: Diagnostics & Validation
Created diagnostic plots:
- **Actual vs Predicted** - Alignment with perfect prediction line
- **Residual Plot** - Error distribution across predictions
- **Residual Distribution** - Histogram of prediction errors
- **Feature Importances** - For tree-based models

---

##  Key Insights

### What the Model Tells Us
1. **Linear Relationship Dominates** - Living area explains 94% of price variance
2. **Strong Model Fit** - R² of 0.93 indicates excellent predictive power
3. **Typical Prediction Error** - ±$39,000 average error
4. **Well-Calibrated** - Residuals normally distributed around zero
5. **No Heteroscedasticity** - Error magnitude consistent across price range

### Actionable Recommendations

1. **For Price Estimation**
   - Use this model for initial valuations
   - Combine with local market adjustments
   - Update annually with new sales data

2. **For Model Improvement**
   - Add neighborhood/location features
   - Include property condition ratings
   - Consider market seasonality

3. **For Production**
   - Implement model monitoring dashboard
   - Track prediction accuracy on new data
   - Retrain quarterly with new properties

---

##  Visualizations

### EDA Visualizations (01_eda_visualizations.png)
- Price distribution and summary statistics
- Scatter plots with trend analysis
- Box plots by categories
- Correlation heatmap
- Feature importance preview

### Model Diagnostics (02_model_diagnostics.png)
- Actual vs predicted scatter plot
- Residual plot showing error patterns
- Histogram of prediction errors
- Feature importance chart

---

##  Methodology

### Regression Approach
This project uses **supervised learning** to predict continuous values (house prices) based on property features.

### Validation Strategy
- **Train-Test Split**: 80-20 split prevents overfitting
- **Cross-Validation**: Ensures robust performance estimates
- **Residual Analysis**: Validates model assumptions

### Error Metrics Interpretation
- **MAE ($39,161)**: Average absolute error in dollars
- **RMSE ($49,185)**: Penalizes larger errors more heavily
- **R² (0.9322)**: Model explains 93.2% of price variation

---

##  Code Features

### Clean, Professional Code
- **Comments**: Every section thoroughly documented
- **Functions**: Reusable, modular design
- **Standards**: PEP 8 compliance
- **Performance**: Efficient numpy/pandas operations

### Example Code Block
```python
# Feature engineering for better model performance
df_model['Age'] = 2020 - df_model['YearBuilt']
df_model['TotalSqft'] = df_model['GrLivArea'] + df_model['GarageArea']

# Train-test split with random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features for linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

##  Dependencies

All dependencies listed in `requirements.txt`:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install with: `pip install -r requirements.txt`

---

## Performance on Different Systems

- Full analysis: ~2 seconds
- Visualizations: ~3 seconds
- Total runtime: <5 seconds
-  Memory usage: <100MB

---



**Questions about the analysis?**
- Review the comprehensive comments in `house_price_analysis.py`
- Check the "Key Insights" section above
- Examine the diagnostic plots for model behavior

**Want to extend this project?**
- Add more features (neighborhood, condition, lot shape)
- Implement hyperparameter tuning
- Deploy as REST API
- Create web interface for predictions

---

##  License

MIT License - Feel free to use for learning and portfolio purposes.

---


---

##  If You Found This Helpful

Please ⭐ star this repository and share it with others learning data science!

---

 
**Status**: Complete & Production-Ready
