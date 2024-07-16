from utils import make_zarr_group, permute_in_chunks, case_of_one
from typing import List, Tuple, Dict, Optional, List
from datetime import time
import pandas as pd
import numpy as np
import zarr
import sys
import os

def merge_zarr_datasets(datasets: List[Dict[str, Tuple[zarr.Array, List[str]]]]) -> Dict[str, Tuple[zarr.Array, List[str]]]:
    '''
    Merge the given datasets into a single dataset.
    Args:
        datasets: The datasets to be merged
    Returns:
        Dict[str, Tuple[zarr.Array, List[str]]]: The merged dataset

    Algorithm:
    - Determine the types of `array_type` and corresponding `array_type` features in all datasets.
    - Generate a hash table for each `array_type` to map features randomly.
    - Create a new zarr group for the merged dataset. Each `array_type` will have a corresponding zarr array. Store on Disk
    - Merge the data into the new dataset.
        - Each zarr array is chunked into 2500 rows. For a max of 2500 rows and 50,000 features, the chunk size is 1GB. 
        - Generate shuffled row numbers in chunks of 2500. This will help us fill the merged dataset randomly but in an optimal fashion -- one chunk at a time.
        - Generate a hash table for each `array_type` and randomly permute the rows. This will help us fill the merged dataset randomly.
        - Iterate through each dataset and merge the data into the new dataset.
            - If `arrary_type` is present, map the features to their respective columns.
            - Pop the row number to be filled in the merged dataset.
            - Fill the merged dataset randomly but in an optimal fashion -- one chunk at a time.
    - Add the merged dataset to the return dictionary.
    - Return the merged dataset.

    Examples:
    >>> dataset1 = {
            "type_A": (zarr.array([[1, 2, 0], [3, 4, 0]]), ["feature1", "feature2", "feature3"]),
            "type_B": (zarr.array([[5, 6], [7, 8]]), ["feature4", "feature5"])
        }
    >>> dataset2 = {
            "type_A": (zarr.array([[7, 8, 9], [10, 11, 12]]), ["feature1", "feature2", "feature4"]),
            "type_C": (zarr.array([[10, 11], [12, 13]]), ["feature6", "feature7"])
        }
    >>> merged_dataset = merge_zarr_datasets([dataset1, dataset2])
    >>> print_func(merged_dataset)
    Array Type: type_C
    feature6	feature7	
    0               0               
    10              11              
    0               0               
    12              13              

    Array Type: type_A
    feature1	feature4	feature3	feature2	
    9               8               7               0               
    0               2               1               0               
    12              11              10              0               
    0               4               3               0               

    Array Type: type_B
    feature5	feature4	
    0               0               
    5               6               
    0               0               
    7               8  
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
        # Recommanded Chunk size is 10MB = 50,000 features * 8 bytes * 25 rows
        num_rows = 10000000 // (num_features*8)
        if num_rows > 5000:
            num_rows = 5000

        chunks = (num_rows, num_features)
        chunks_dict[array_type] = chunks
        print(f'Array Type: {array_type} | Num Features: {num_features} | Chunk Size: {chunks}')
        
        # A new merged array for each array type
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
            # print('\tRunning for array type:', array_type)
            if array_type in dataset.keys():
                source_array, source_features = dataset[array_type]
                # Map the features to their respective columns
                for rows in range(in_rows):
                    print(f'\t\tRow: {rows}/{in_rows}', end='\r')
                    # Get the row number to be filled in the merged dataset -- we are ensuring that we fill the merged dataset randomly but in optimal fashion -- one chunk at a time.
                    select_row = row_hash[array_type].pop(0)
                    # Order source features based on the feature_sets[array_type]
                    source_features_ordered = [feature_mappings[array_type][feature] for feature in source_features]
                    source_features_data = source_array[rows, :]
                    # Main Magic Happens Here
                    feed = np.zeros(len(feature_mappings[array_type]), dtype=np.int64)
                    # Assign the source data to the correct columns
                    feed[source_features_ordered] = source_features_data
                    merged_arrays[array_type][select_row, :] = feed
            # If the array type is not present in the dataset, fill the merged dataset with zeros by default
    
    return_merged_array = {}
    for array_type in all_array_types:
        return_merged_array[array_type] = (merged_arrays[array_type], list(feature_mappings[array_type]))
    return return_merged_array


def verify_randomization(original_datasets: List[Dict[str, Tuple[zarr.Array, List[str]]]], merged_dataset: Dict[str, Tuple[zarr.Array, List[str]]]) -> bool:
    '''
    Verify that the merged dataset is a randomization of the original datasets.
    Args:
        original_datasets: The original datasets to be merged
        merged_dataset: The merged dataset
    Returns:
        bool: True if the merged dataset is a randomization of the original datasets, False otherwise
    Algorithm:
    - Part 1: Collect all array types, their features, and calculate total rows
    - Part 2: Create a Feature Mapping (Hash Table) for each array type
    - Part 3: Verify the randomization
        - For each array type, check if the rows from the original datasets are present in the merged dataset
    - Return True if the merged dataset is a randomization of the original datasets, False otherwise
    '''
    # Part 1: Collect all array types, their features, and calculate total rows
    all_array_types = set()
    feature_sets = {}
    total_rows = merged_dataset[list(merged_dataset.keys())[0]][0].shape[0]
    for array_type, (array, features) in merged_dataset.items():
        all_array_types.add(array_type)
        if array_type not in feature_sets:
            feature_sets[array_type] = set()
        feature_sets[array_type].update(features)
    
    # Part 2: Create a Feature Mapping (Hash Table) for each array type
    feature_mappings = {
        array_type: {feature: idx for idx, feature in enumerate(features)}
        for array_type, features in feature_sets.items()
    }

    # Part 3: Verify the randomization
    for array_type in all_array_types:
        merged_array, merged_features = merged_dataset[array_type]
        for dataset in original_datasets:
            original_arr_types = set(dataset.keys())
            if array_type not in original_arr_types:
                continue
            original_array, original_features = dataset[array_type]
            original_features_mapped = [feature_mappings[array_type][feature] for feature in original_features]
            original_features_mapped = np.sort(original_features_mapped)
            merged_array_feature_indexed = merged_array[:, original_features_mapped]
            original_array_ordered = original_array[:, np.argsort(original_features_mapped)]
            for row in range(len(original_array_ordered)):
                original_row = original_array_ordered[row]
                if original_row not in merged_array_feature_indexed:
                    print(f'Row {original_row} from {array_type} not found in the merged dataset {merged_array_feature_indexed}')
                    return False
    return True


'''
Possible Issues:
- The input datasets are chunked into some other size. Need to rechunk.
- The input datasets are not chunked. Need to chunk.
'''