# Advanced Regression - House Prices

This project is for CSC-340 Data Science Course. A comprehensive data science project implementing stacked ensemble models with hyperparameter optimization for predicting house prices using the Kaggle House Prices dataset.

## 📊 Project Overview

This repository contains an end-to-end machine learning pipeline for predicting residential property prices in Ames, Iowa. The project demonstrates advanced techniques in feature engineering, hyperparameter tuning, and ensemble learning to achieve optimal prediction accuracy.


## 🔬 Methodology

### Phase 1: Exploratory Data Analysis
- Comprehensive missing value analysis and treatment
- Distribution analysis of target variable (SalePrice)
- Correlation heatmaps identifying top predictive features
- Categorical and numerical feature relationships with price
- Visual analysis of key drivers (OverallQual, GrLivArea, Neighborhood)

**Key Findings:**
- OverallQual is the strongest predictor (correlation > 0.79)
- Location (Neighborhood) heavily influences pricing
- GrLivArea shows strong positive correlation with price

### Phase 2: Baseline Model Development
Three baseline models were implemented:
1. **Random Forest Regressor** (n_estimators=300, max_depth=12)
2. **LightGBM Regressor** (n_estimators=1000, learning_rate=0.05)
3. **XGBoost Regressor** (n_estimators=1000, learning_rate=0.05)

### Phase 3: Advanced Stacking Ensemble

#### Feature Engineering
Created 15+ engineered features:
- **Age-based:** PropertyAge, HasRemodeled
- **Area aggregations:** TotalSF, TotalBath
- **Amenity indicators:** HasPool, HasBasement, HasFireplace, HasGarage
- **Quality metrics:** OverallGrade, QualityArea
- **Temporal categorization:** YrSold_cat, MoSold_cat

#### Preprocessing Pipeline
- **Categorical features:** Imputation + One-Hot Encoding
- **Numerical features:** Mean imputation + StandardScaler
- **Automated pipeline** using scikit-learn's ColumnTransformer

#### Model Optimization (Optuna)
Hyperparameter tuning performed for each base model:

| Model | Best RMSE | Key Parameters |
|-------|-----------|----------------|
| Ridge | 0.11234 | alpha=6.58 |
| ElasticNet | 0.11189 | alpha=0.0012, l1_ratio=0.26 |
| KernelRidge | 0.11567 | kernel=polynomial, degree=2 |
| XGBoost | 0.12456 | max_depth=4, lr=0.024 |
| LightGBM | 0.12134 | num_leaves=278, lr=0.062 |

#### Stacked Meta-Model
- **Architecture:** ElasticNet meta-learner on out-of-fold predictions
- **Final RMSE:** Improved over individual base models
- **Validation:** 10-fold cross-validation

## 👥 Contributors

Data Science Project Team : Akari Kyaw Thein, Ant Bone Kyaw, Daniel Bawm Ying, Thaw Zin Moe Myint.

## 🙏 Acknowledgments

- Kaggle House Prices Competition
- scikit-learn and Optuna communities
- Ames, Iowa Housing Dataset

---