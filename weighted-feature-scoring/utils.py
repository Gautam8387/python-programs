from scipy.sparse import csr_matrix, load_npz
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import pandas as pd
import numpy as np
import random
import time
import zarr

def load_data(graph_csr_path:str, partition_path:str, zarr_array_path:str) -> Tuple[csr_matrix, np.ndarray, zarr.Array]:
    """
    Load the data from the given paths.
    -------
    Args:
        - graph_csr_path: Path to the graph csr matrix
        - partition_path: Path to the partition file
        - zarr_array_path: Path to the zarr array
    Returns:
        - graph_csr: csr_matrix: The graph csr matrix
        - partition: np.ndarray: The partition array
        - feature_matrix: zarr.Array: The feature matrix
    """
    partition = np.load(partition_path)
    partition -= partition.min()
    graph_csr = load_npz(graph_csr_path)
    feature_matrix = zarr.open(zarr_array_path, mode='a')
    return graph_csr, partition, feature_matrix

def make_graph(graph_csr:csr_matrix, partition:np.ndarray) -> nx.Graph:
    """
    Create a graph from the csr matrix
    """
    graph = nx.from_scipy_sparse_array(graph_csr)
    node_group_map = dict(zip(range(len(partition)), partition))
    nx.set_node_attributes(graph, node_group_map, 'group')
    # Graph Data
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    print("Density:", nx.density(graph))
    print("Number of connected components:", nx.number_connected_components(graph))
    print("First 10 edge weights:", list(graph.edges(data=True))[:10])
    return graph


def zarr_properties(zarr_array:zarr.Array) -> Dict[str, Any]:
    """
    Get the properties of the zarr array.
    -------
    Args:
        - zarr_array: zarr.Array: The zarr array
    Returns:
        - properties: Dict[str, Any]: The properties of the zarr array
    """
    n, f = zarr_array.shape
    chunkks = zarr_array.chunks
    row_chunks = (n // chunkks[0]) + 1
    col_chunks = (f // chunkks[1]) + 1
    properties =  {
        'n': n,
        'f': f,
        'row_chunks': row_chunks,
        'col_chunks': col_chunks,
        'chunks': chunkks,
        'nchunks': zarr_array.nchunks,
        'dtype': zarr_array.dtype
    }
    return properties


def group_explanation(sum_deg_s:pd.Series, ranked_features:Dict[int, pd.Series]) -> None:
    """
    Provide an explanation of the groups based on the relationships and the ranked features.
    """
    print(f'\nTop 5 Groups with Highest Sum of Similarity')
    print(sum_deg_s.head())
    print(f'\nTop 5 Features for Each Group')
    for group, features in ranked_features.items():
        print(f'\nGroup {group}')
        print(features.head())