from utils import load_data, make_graph, zarr_properties
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import pickle as pkl
import pandas as pd
import numpy as np
import argparse
import psutil
import time
import zarr
import math
import os

def calculate_scores(feature_matrix:zarr.Array, partition:np.ndarray) -> pd.DataFrame:
    """
    Calculate the scores for the given feature matrix, partition and graph csr matrix.
    -------
    Args:
        - feature_matrix: zarr.Array: The feature matrix
        - partition: np.ndarray: The partition array
        - graph_csr: csr_matrix: The graph csr matrix
    Returns:
        - scores: pd.DataFrame: The scores dataframe
    -------
    - Iterate over the chunks (in a block-wise manner) of the feature matrix and calculate the scores.
    - For each row chunk, iterate over the column chunks and calculate the ranks.
    - For each column chunk, calculate the ranks and group them by the partition.
    """
    n, f = feature_matrix.shape
    row_chunks = math.ceil(n / feature_matrix.chunks[0])
    col_chunks = math.ceil(f / feature_matrix.chunks[1])
    # Initialize the total ranks as a dictionary. Each key is a column chunk and the value is None.
    # Each row chunk has a total rank for all its column chunks.
    total_ranks = {}
    for col in range(col_chunks):
        total_ranks[col] = None
    # Iterate over the row chunks
    for row in range(row_chunks):
        # Iterate over the column chunks for each row chunk
        row_idx_start = row * feature_matrix.chunks[0]
        row_idx_end = min(n, (row+1) * feature_matrix.chunks[0])
        group_chunk = pd.Series(partition[row_idx_start:row_idx_end]) # Get the partition for the row chunk
        for col in range(col_chunks):
            chunk = feature_matrix.get_block_selection((row, col))  # Get the chunk - block-wise to ensure memory efficiency
            chunk_df = pd.DataFrame(chunk) 
            chunk_ranks = chunk_df.rank(method='dense', ascending=False) # Rank the chunk using dense ranking. Equal values are assigned the same rank.
            chunk_ranks_grouped = chunk_ranks.groupby(group_chunk).sum() # Group the ranks by the partition
            missing_partitions = set(partition) - set(chunk_ranks_grouped.index) # Check if all the partitions are present in the grouped ranks
            if missing_partitions: # If not, add the missing partitions and assign 0
                for p in missing_partitions:
                    chunk_ranks_grouped.loc[p] = 0
            
            if total_ranks[col] is None: 
                total_ranks[col] = chunk_ranks_grouped
            else:
                total_ranks[col] += chunk_ranks_grouped
    
    # Concatenate the total ranks for each column chunk into a single dataframe and calculate the mean ranks
    concat_df = None
    for key, df, in total_ranks.items():
        df = df.sort_index()
        concat_df = pd.concat([concat_df, df], axis=1)
    
    mean_ranks = concat_df / n
    score = mean_ranks.reindex(set(partition))
    score = score / score.sum() # Normalize the scores
    score_pd = pd.DataFrame(score)
    score_pd.columns = [f'{i}' for i in range(len(score_pd.columns))]
    # For a score_pd with 1000 groups and 1000 features, the memory usage is around 7.63 MB (7812.5 KB)
    return score_pd

def build_graph_relationship(graph:nx.Graph, partition:np.ndarray) -> pd.DataFrame:
    """
    Analyze the relationships between the groups in the graph.
    -------
    Args:
        - graph: nx.Graph: The graph
        - partition: np.ndarray: The partition array
    Returns:
        - sim_matrix: pd.DataFrame: The similarity matrix between the groups
    -------
    - Create a similarity matrix between the groups based on the edge weights in the graph.
    - The similarity between two groups is the sum of the edge weights between the nodes in the two groups.
    """
    groups = set(partition) # Get the unique groups
    # Initialize the similarity matrix between the groups - a dictionary of dictionaries
    sim_matrix = {g: {h: 0 for h in groups} for g in groups}
    # Add self-relationships
    for g in groups:
        sim_matrix[g][g] = 1
    # Add the edge weights to the similarity matrix
    for u, v, data in graph.edges(data=True):
        # If the edge references a node not in the partition, skip it
        if u >= len(partition) or v >= len(partition):
            print(f"Warning: Edge ({u}, {v}) references a node not in the partition.")
            continue

        # Assign the edge weight to the similarity between the groups
        g_u, g_v = partition[u], partition[v]
        sim_matrix[g_u][g_v] += data.get('weight', 0) # Use the get method to get the weight. If not present, assign 0
        sim_matrix[g_v][g_u] += data.get('weight', 0)
    
    sim_matrix = pd.DataFrame(sim_matrix)
    return sim_matrix

def graph_relationship_analysis(sim_matrix:pd.DataFrame, score:pd.DataFrame) -> pd.Series:
    """
    Analyze the relationships between the groups in the graph.
    -------
    Args:
        - sim_matrix: pd.DataFrame: The similarity matrix between the groups
        - score: pd.DataFrame: The scores dataframe
    Returns:
        - adjusted_scores: pd.DataFrame: The adjusted scores dataframe
    """
    print(f'Similarity Matrix Shape: {sim_matrix.shape}')
    print(f'\nPrtinting Row Wise Sum of Similarity Matrix')
    sum_deg = {}
    for idx, row in sim_matrix.iterrows():
        sum = 0
        for i in range(len(row)):
            if i != idx:
                sum += row[i]
        sum_deg[idx] = sum
        print(f'\tRow {idx} sum: {sum}')
    sum_deg_s = pd.Series(sum_deg)
    sum_deg_s.sort_values(ascending=False, inplace=True)
    print(f'\nTop 5 Groups with Highest Sum of Similarity')
    print(sum_deg_s.head())
    return sum_deg_s

def adjusted_scoring(scores:pd.DataFrame, sim_matrix:pd.DataFrame) -> pd.DataFrame:
    """
    Adjust the scores based on the relationships between the groups.
    -------
    Args:
        - scores: pd.DataFrame: The scores dataframe
        - sim_matrix: pd.DataFrame: The similarity matrix between the groups
        - partition: np.ndarray: The partition array
    Returns:
        - adjusted_scores: pd.DataFrame: The adjusted scores dataframe
    """
    adjusted_scores = scores.copy()
    for group in adjusted_scores.index:
        for other_group in adjusted_scores.index:
            if group == other_group: # Skip the same group
                continue
            # print(f'Group {group} vs Group {other_group}')
            # Similarity between the groups
            similarity = sim_matrix.loc[group, other_group]
            # Adjust the scores based on the similarity and the scores of the other group
            adjusted_scores.loc[group] += similarity * scores.loc[other_group]
    # Normalize the scores
    adjusted_scores = adjusted_scores / adjusted_scores.sum()
    return adjusted_scores

def visualize_change(scores:pd.DataFrame, adjusted_scores:pd.DataFrame, path:str) -> None:
    """
    Visualize the change in the scores after adjustment.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(scores, cmap='viridis', ax=ax[0])
    ax[0].set_title('Original Scores', fontsize=14)
    sns.heatmap(adjusted_scores, cmap='viridis', ax=ax[1])
    ax[1].set_title('Adjusted Scores', fontsize=14)
    plt.suptitle('Change in Scores After Adjustment', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{path}/change_in_scores.png')
    plt.close()

def rank_features(adjusted_scores:pd.DataFrame) -> Dict[int, pd.Series]:
    """
    Rank the features based on the adjusted scores of the groups.
    """
    ranked_features = {}
    for group in adjusted_scores.index:
        ranked_features[group] = adjusted_scores.loc[group].sort_values(ascending=False)
    return ranked_features

def main():
    """
    Main function to run the feature scoring pipeline.
    """
    # Arguments
    parser = argparse.ArgumentParser(description='Feature Scoring Pipeline')
    parser.add_argument('--graph_csr', type=str, help='Path to the Graph CSR Matrix (.npz) file', required=True)
    parser.add_argument('--group', type=str, help='Path to the Group Array (.npy) file', required=True)
    parser.add_argument('--feature_matrix', type=str, help='Path to the Feature Matrix (.zarr) file', required=True)    
    args = parser.parse_args()
    os.makedirs('output', exist_ok=True)
    # Load the data
    print("Loading Data...")
    graph_csr, partition, feature_matrix = load_data(args.graph_csr, args.group, args.feature_matrix)
    # Create a graph from the csr matrix
    print("Creating Graph...")
    graph = make_graph(graph_csr, partition)
    # Get the properties of the zarr array
    properties = zarr_properties(feature_matrix)
    print(f'Feature Matrix Properties: {properties}')
    start_time = time.time()
    
    # Calculate the scores for the feature matrix
    print("Calculating Scores...")
    scores = calculate_scores(feature_matrix, partition)
    scores.to_csv('output/original-scores.csv')
    print(f'Original Scores saved as original-scores.csv')
    
    # Build the graph relationship
    print("Building Graph Relationship...")
    sim_matrix = build_graph_relationship(graph, partition)
    sim_matrix.to_csv('output/similarity-matrix.csv')
    print(f'Similarity Matrix saved as similarity-matrix.csv')
    
    # Analyze the relationships between the groups
    print("Analyzing Relationships...")
    sum_deg_s = graph_relationship_analysis(sim_matrix, scores)
    pkl.dump(sum_deg_s, open('output/sum_deg_s.pkl', 'wb'))
    print(f'Sum of Degrees saved as sum_deg_s.pkl')
    
    # Adjust the scores based on the relationships
    print("Adjusting Scores...")
    adjusted_scores = adjusted_scoring(scores, sim_matrix)
    adjusted_scores.to_csv('output/adjusted-scores.csv')
    print(f'Adjusted Scores saved as adjusted-scores.csv')

    # Visualize the change in the scores
    print("Visualizing Change...")
    visualize_change(scores, adjusted_scores, 'output')
    print(f'Visualization saved as change_in_scores.png')
    
    # Rank the features based on the adjusted scores
    print("Ranking Features...")
    ranked_features = rank_features(adjusted_scores)
    pkl.dump(ranked_features, open('output/ranked_features.pkl', 'wb'))
    print(f'Ranked Features saved as ranked_features.pkl')
    
    print(f'Feature Scoring Pipeline completed in {time.time() - start_time:.2f} seconds.')
    # Peak Memory Usage
    print(f"Peak memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    # IO Statistics - Total Data Read, Number of Read Operations, Total Data Written, Number of Write Operations
    print(f"IO Statistics: {psutil.disk_io_counters()}")
    

if __name__ == '__main__':
    main()