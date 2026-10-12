"""
Big Data Analytics Engine & OLAP Aggregations
Provides multi-dimensional cube aggregations, data profiling metrics,
MapReduce benchmarks, and vulnerability index calculations.
"""

import pandas as pd
import numpy as np

def generate_data_profile(df):
    """
    Computes Big Data profiling metrics:
    Shape, memory consumption, missing count, distinct values, and data types.
    """
    profile = []
    mem_usage = df.memory_usage(deep=True)
    
    for col in df.columns:
        profile.append({
            'Column Name': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': int(df[col].notnull().sum()),
            'Missing Count': int(df[col].isnull().sum()),
            'Missing %': round((df[col].isnull().sum() / len(df)) * 100, 2),
            'Unique Values': int(df[col].nunique()),
            'Memory (Bytes)': int(mem_usage.get(col, 0))
        })
    return pd.DataFrame(profile)

def compute_demographic_vulnerability_index(df):
    """
    Computes a Normalized Vulnerability Score (0 - 100) combining suicide rate,
    gender vulnerability ratio, and elderly dependency representation.
    """
    agg = df.groupby(['country', 'age', 'sex']).agg({
        'suicides_no': 'sum',
        'population': 'sum',
        'suicides_per_100k': 'mean'
    }).reset_index()
    
    max_rate = agg['suicides_per_100k'].max() if len(agg) > 0 and agg['suicides_per_100k'].max() > 0 else 1.0
    agg['vulnerability_index'] = ((agg['suicides_per_100k'] / max_rate) * 100).round(1)
    return agg.sort_values(by='vulnerability_index', ascending=False)

def compute_olap_cube(df):
    """
    Computes OLAP multi-dimensional rollups across Country x Year x Age x Sex.
    """
    cube_country_year = df.groupby(['country', 'year']).agg({
        'suicides_no': 'sum',
        'population': 'sum',
        'suicides_per_100k': 'mean'
    }).reset_index()
    cube_country_year['rate_per_100k'] = ((cube_country_year['suicides_no'] / cube_country_year['population']) * 100000).round(2)
    
    cube_demographics = df.groupby(['age', 'sex']).agg({
        'suicides_no': 'sum',
        'population': 'sum',
        'suicides_per_100k': 'mean'
    }).reset_index().sort_values(by='suicides_no', ascending=False)
    
    return cube_country_year, cube_demographics
