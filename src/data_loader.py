"""
Data Ingestion and Big Data Processing Engine
Handles data loading, chunking simulation, and schema validation.
"""

import os
import pandas as pd
import numpy as np
import time

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "master.csv")

def load_raw_data(filepath=None):
    """
    Load raw CSV data.
    """
    path = filepath if filepath and os.path.exists(filepath) else DEFAULT_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path)
    return df

def simulate_mapreduce_processing(df, chunk_size=10):
    """
    Simulates a Big Data MapReduce aggregation pipeline over chunks.
    Demonstrates distributed computing concepts (Map -> Shuffle & Sort -> Reduce)
    on multi-dimensional analytics.
    """
    start_time = time.time()
    num_chunks = int(np.ceil(len(df) / chunk_size))
    
    # Map Step: Calculate intermediate stats per chunk
    mapped_records = []
    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        for _, row in chunk.iterrows():
            key = (row.get('country', 'Unknown'), row.get('sex', 'Unknown'))
            val = {
                'suicides': row.get('suicides_no', 0),
                'population': row.get('population', 0),
                'count': 1
            }
            mapped_records.append((key, val))
            
    # Shuffle & Sort Step: Group by key
    shuffled = {}
    for key, val in mapped_records:
        if key not in shuffled:
            shuffled[key] = []
        shuffled[key].append(val)
        
    # Reduce Step: Aggregate per group
    reduced = []
    for key, val_list in shuffled.items():
        total_suicides = sum(v['suicides'] for v in val_list)
        total_pop = sum(v['population'] for v in val_list)
        rate = (total_suicides / total_pop * 100000) if total_pop > 0 else 0
        reduced.append({
            'country': key[0],
            'sex': key[1],
            'total_suicides': total_suicides,
            'total_population': total_pop,
            'suicide_rate_per_100k': round(rate, 2),
            'records_processed': sum(v['count'] for v in val_list)
        })
        
    execution_time = (time.time() - start_time) * 1000 # in ms
    summary_df = pd.DataFrame(reduced)
    
    metrics = {
        'total_rows': len(df),
        'num_chunks': num_chunks,
        'execution_time_ms': round(execution_time, 2),
        'unique_groups': len(reduced)
    }
    
    return summary_df, metrics
