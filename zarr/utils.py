from typing import List, Tuple, Dict, Optional, List
from datetime import time
import pandas as pd
import numpy as np
import zarr
import sys
import os

def check_available_name() -> str:
    '''
    Check the available name for the new merged array
    Args:
        None
    Returns:
        str: the name of the new merged array
    '''
    folders = os.listdir()
    # check folder name starting with merged_arrays
    folders = [folder for folder in folders if folder.startswith("merged_arrays")]
    folders.sort()
    # return the next name
    return f"merged_arrays_{len(folders)+1}.zarr"

def print_func(merged_arrays: Dict[str, Tuple[zarr.Array, List[str]]]) -> None:
    '''
    Print the merged arrays
    Args:
        merged_arrays: The merged arrays
    Returns:
        None
    '''    
    for key, val in merged_arrays.items():
        features = val[1]
        print(f'Array Type: {key}')
        # print features
        for feat in features:
            print(feat, end='\t')
        print()
        # print data
        for row in val[0]:
            for col in row:
                print(col,' '*(10-len(str(col))), end=' '*5)
            print()
        print()

def permute_in_chunks(size:int, chunk_size:int, seed:int=42) -> List[int]:
    '''
    Permutate an array in chunks of a given size.
    Args:
        size: The size of the array to be permuted
        chunk_size: The size of the chunks to permute
    Returns:
        A permuted array of the given size
    Examples:
    >>> permute_in_chunks(10, 3)
    array([2, 0, 1, 5, 4, 3, 9, 7, 8, 6])
    '''
    np.random.seed(seed)
    arr = np.arange(size)
    permuted_arr = np.empty(size, dtype=int)
    
    for start in range(0, size, chunk_size):
        end = start + chunk_size
        permuted_arr[start:end] = np.random.permutation(arr[start:end])

    return list(permuted_arr)

def make_zarr_group(total_rows:int, all_array_types:List[str], feature_mappings:Dict[str, List[str]], seed:int=42) -> zarr.Group:
    '''
    Create a zarr group with the given parameters.
    Args:
        total_rows: The total number of rows in the group
        all_array_types: A list of all array types in the group
        feature_mappings: A dictionary mapping array types to their features
        seed: The seed for the random number generator
    Returns:
        group: A zarr group with the given parameters
    '''
    group = zarr.group(store=zarr.DirectoryStore(f'{check_available_name()}'))
    group.attrs['total_rows'] = total_rows
    group.attrs['all_array_types'] = list(all_array_types)
    group.attrs['feature_mappings'] = feature_mappings
    group.attrs['seed'] = seed
    group.attrs['datetime'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    return group


def case_of_one(dataset:Dict[str, Tuple[zarr.Array, List[str]]]) -> Dict[str, Tuple[zarr.Array, List[str]]]:
    '''
    Return the dataset with randomised rows and features.
    Args:
        dataset: A dictionary of arrays to merge
    Returns:
        merged_array: A dictionary containing the merged array
    '''
    total_rows = dataset[list(dataset.keys())[0]][0].shape[0]
    shuffled_rows = permute_in_chunks(total_rows, 2500)
    row_hash = {array_type: shuffled_rows.copy() for array_type in dataset.keys()}
    feature_mappings = {}

    for array_type, features in dataset.items():
        permuted_features = np.random.permutation(features[1])
        feature_mappings[array_type] = {feature: idx for idx, feature in enumerate(permuted_features)}

    merged_arrays = {}
    merged_arrays_zarr = make_zarr_group(total_rows, list(dataset.keys()), feature_mappings)
    for array_type in dataset.keys():
        merged_arrays[array_type] = merged_arrays_zarr.zeros(shape=(total_rows, len(feature_mappings[array_type])), dtype=np.int64, chunks=(10000, len(feature_mappings[array_type])), name=f'{array_type}.zarr', overwrite=True, default_fill_value=0)
        merged_arrays[array_type].attrs['features'] = list(feature_mappings[array_type])

        source_array, source_features = dataset[array_type]
        for row in range(total_rows):
            select_row = row_hash[array_type].pop(0)
            for feature in source_features:
                val = source_array.vindex[row, source_features.index(feature)]
                merged_arrays[array_type].vindex[select_row, feature_mappings[array_type][feature]] = val

    return_merged_array = {array_type: (merged_arrays[array_type], list(feature_mappings[array_type])) for array_type in dataset.keys()}
    return return_merged_array