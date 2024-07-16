from utils import make_zarr_group, permute_in_chunks, case_of_one, do_batch
from typing import List, Tuple, Dict, Optional, List
from datetime import time
import pandas as pd
import numpy as np
import zarr
import sys
import os

def merge_zarr_datasets(datasets: List[Dict[str, Tuple[zarr.Array, List[str]]]]) -> Dict[str, Tuple[zarr.Array, List[str]]]:
    '''
    Merge multiple zarr datasets into a single dataset
    Args:
        datasets: A list of dictionaries where each dictionary contains the array type as key and a tuple of the array and its features as value
    Returns:
        return_merged_array: A dictionary containing the array type as key and a tuple of the merged array and its features as value
    '''
    seed = 42
    np.random.seed(seed)
    
    if not datasets: # Case for empty datasets
        raise ValueError("No datasets provided")
    if len(datasets) == 1: # Case for one dataset -- no need to merge, return a randomised dataset 
        return_merged_array = case_of_one(datasets[0])
        return return_merged_array

    # Part 1: Collect all array types, their features, and calculate total rows
    all_array_types = set()
    feature_sets = {}
    total_rows = 0
    for dataset in datasets:
        # All `array_types` within a dataset have the same number of rows and the same row order. Loading the actual data can be expensive, so we only load the first dataset. 
        # Possible: Create an iterator to avoid loading all datasets. total_rows += next(iter(dataset.values()))[0].shape[0]
        total_rows += dataset[list(dataset.keys())[0]][0].shape[0]
        # Make a single pass through the dataset to collect all array types and features
        for array_type, (array, features) in dataset.items(): 
            all_array_types.add(array_type)
            if array_type not in feature_sets:
                feature_sets[array_type] = set()
            feature_sets[array_type].update(features)

    # Part 2: Create a Feature Mapping (Hash Table) for each array type
    # This will be useful when we need to map features to their respective columns
    feature_mappings = {} 
    for array_type, features in feature_sets.items():
        # Randomly permute the features
        permuted_features = np.random.permutation(list(features)) 
        feature_mappings[array_type] = {feature: idx for idx, feature in enumerate(permuted_features)}

    # Part 3: Create a new dataset with the merged data
    merged_arrays = {}
    merged_arrays_zarr = make_zarr_group(total_rows, all_array_types, feature_mappings)
    chunks_dict = {}
    
    for array_type in all_array_types:
        num_features = len(feature_sets[array_type])
        ram_size_allocated = 1*1024*1024*1024 # (1GB)
        num_rows = ram_size_allocated // (num_features*8) # 8 for int64
        if num_rows > 2500:
            num_rows = 2500

        chunks = (num_rows, num_features)
        chunks_dict[array_type] = chunks
        print(f'Array Type: {array_type} | Num Features: {num_features} | Chunk Size: {chunks}')
        
        # A new merged array for each array type in a group
        merged_arrays[array_type] = merged_arrays_zarr.zeros(shape=(total_rows, num_features), dtype=np.int64, name=f'{array_type}.zarr', overwrite=True, chunks=chunks)
        merged_arrays[array_type].attrs['features'] = list(feature_sets[array_type])

    # Part 4: Merge the data
    # We will create a hash for each row which is filled -- this will help us when filling the merged dataset randomly
    # Explicitly copy the list to avoid reference issues & Maintain randomization across all rows
    row_hash = {arr_type: permute_in_chunks(total_rows, chunks_dict[arr_type][0]).copy() for arr_type in all_array_types}

    for dataset in datasets:
        print("Starting new dataset")
        in_rows = dataset[list(dataset.keys())[0]][0].shape[0] # Number of rows in current dataset
        # print(in_rows)
        for array_type in all_array_types:
            print('\tRunning for array type:', array_type)
            if array_type in dataset.keys():
                source_array, source_features = dataset[array_type]
                # Map the features to their respective columns

                # Implement Batching and chunking
                num_batches = in_rows // chunks_dict[array_type][0]
                chunk_rows = chunks_dict[array_type][0]
                if num_batches == 0:
                    print("\t\tNo Batches")
                    feed, feed_row = do_batch(None, None, in_rows, row_hash, array_type, feature_mappings, source_array, source_features)
                    merged_arrays[array_type][feed_row, :] = feed
                else:
                    print(f'\tNumber of Batches: {num_batches} | Chunk Rows: {chunk_rows}')
                    for batch in range(num_batches):
                        start_row = batch*chunk_rows
                        end_row = (batch+1)*chunk_rows
                        feed, feed_row = do_batch(start_row, end_row, in_rows, row_hash, array_type, feature_mappings, source_array, source_features)
                        merged_arrays[array_type][feed_row, :] = feed
    
    return_merged_array = {}
    for array_type in all_array_types:
        return_merged_array[array_type] = (merged_arrays[array_type], list(feature_mappings[array_type]))
    return return_merged_array