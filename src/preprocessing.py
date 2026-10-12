"""
Big Data Analytics - Data Preprocessing Pipeline
Provides data cleaning, missing value imputation, outlier detection,
feature engineering, categorical encoding, and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

AGE_ORDER_MAP = {
    '5-14 years': 0,
    '15-24 years': 1,
    '25-34 years': 2,
    '35-54 years': 3,
    '55-74 years': 4,
    '75+ years': 5
}

def clean_data(df):
    """
    Standardizes column names, trims text fields, and casts numeric types.
    """
    clean_df = df.copy()
    
    # Rename columns to clean identifiers
    clean_df.columns = (
        clean_df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('/', '_')
        .str.replace('$', '')
        .str.replace('(', '')
        .str.replace(')', '')
    )
    
    # Standardize column names mapping
    rename_dict = {
        'suicides_100k_pop': 'suicides_per_100k',
        'suicides_100k_pop': 'suicides_per_100k',
        'hdi_for_year': 'hdi',
        'gdp_for_year_': 'gdp_year',
        'gdp_for_year': 'gdp_year',
        'gdp_per_capita_': 'gdp_per_capita',
        'gdp_per_capita': 'gdp_per_capita',
        'country-year': 'country_year'
    }
    clean_df = clean_df.rename(columns=rename_dict)
    
    # Clean text columns
    for col in ['country', 'sex', 'age', 'generation']:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str).str.strip()
            
    # Clean GDP if formatted with commas
    if 'gdp_year' in clean_df.columns:
        if clean_df['gdp_year'].dtype == object:
            clean_df['gdp_year'] = clean_df['gdp_year'].astype(str).str.replace(',', '').str.replace('$', '').astype(float)
        else:
            clean_df['gdp_year'] = clean_df['gdp_year'].astype(float)
            
    # Ensure numeric columns
    num_cols = ['year', 'suicides_no', 'population', 'suicides_per_100k', 'gdp_per_capita']
    for col in num_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            
    # Impute HDI or missing values
    if 'hdi' in clean_df.columns:
        if clean_df['hdi'].isnull().any():
            clean_df['hdi'] = clean_df['hdi'].fillna(clean_df['hdi'].median())
            
    return clean_df

def detect_outliers_iqr(df, column='suicides_no'):
    """
    Calculates Interquartile Range (IQR) outliers.
    """
    if column not in df.columns:
        return df, 0, 0, 0
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = max(0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound, len(outliers)

def engineer_features(df):
    """
    Applies feature engineering:
    - Age ordinal encoding
    - Gender binary flag
    - Risk classification category (Low, Medium, High)
    - Suicide rate verification
    - Socio-demographic interaction term
    """
    df_feat = df.copy()
    
    # Verify/calculate rate per 100k
    df_feat['calculated_rate_100k'] = (df_feat['suicides_no'] / df_feat['population']) * 100000
    df_feat['calculated_rate_100k'] = df_feat['calculated_rate_100k'].round(2)
    
    # Age ordinal mapping
    df_feat['age_num'] = df_feat['age'].map(AGE_ORDER_MAP).fillna(2).astype(int)
    
    # Sex binary encoding (1 = male, 0 = female)
    df_feat['is_male'] = (df_feat['sex'].str.lower() == 'male').astype(int)
    
    # Categorize Risk Level based on suicide rate per 100k
    # Thresholds: Low (< 20), Medium (20 - 45), High (> 45)
    conditions = [
        df_feat['suicides_per_100k'] < 20.0,
        (df_feat['suicides_per_100k'] >= 20.0) & (df_feat['suicides_per_100k'] <= 45.0),
        df_feat['suicides_per_100k'] > 45.0
    ]
    choices = ['Low Risk', 'Moderate Risk', 'High Risk']
    df_feat['risk_category'] = np.select(conditions, choices, default='Moderate Risk')
    
    label_map = {'Low Risk': 0, 'Moderate Risk': 1, 'High Risk': 2}
    df_feat['risk_label'] = df_feat['risk_category'].map(label_map)
    
    return df_feat

def prepare_ml_data(df_feat):
    """
    Prepares X and y for Regression and Classification tasks.
    Performs One-Hot Encoding and Scaling.
    """
    # Features to use
    categorical_cols = ['country', 'sex', 'age']
    numeric_cols = ['year', 'population']
    
    # Include HDI / GDP if variance exists
    for col in ['hdi', 'gdp_year', 'gdp_per_capita']:
        if col in df_feat.columns and df_feat[col].nunique() > 1:
            numeric_cols.append(col)
            
    # One-hot encode categoricals while preserving original columns
    dummies = pd.get_dummies(df_feat[['country', 'sex', 'age']], drop_first=False).astype(float)
    df_encoded = pd.concat([df_feat, dummies], axis=1)
    
    dummy_cols = [c for c in dummies.columns]
    feature_cols = dummy_cols + numeric_cols
    
    X = df_encoded[feature_cols].astype(float).copy()
    y_reg = df_encoded['suicides_per_100k'].astype(float).copy()
    y_clf = df_encoded['risk_label'].astype(int).copy()
    
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return {
        'X': X,
        'X_scaled': X_scaled,
        'y_reg': y_reg,
        'y_clf': y_clf,
        'feature_cols': feature_cols,
        'numeric_cols': numeric_cols,
        'scaler': scaler,
        'encoded_df': df_encoded
    }
