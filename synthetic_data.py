# -*- coding: utf-8 -*-
"""

@author: shuai
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.spatial import KDTree, distance


def plot_fov(df, width=760, height=760, save_file = "", resolution=400):
    plt.figure(figsize=(8, 8), dpi=resolution)
    

    colors = {'Type A': 'red', 'Type B': 'blue'} 
    
    for cell_type, group in df.groupby('cell_type'):
        plt.scatter(
            group['X'], 
            group['Y'], 
            label=cell_type, 
            color=colors[cell_type],
            s=40,         
            edgecolor='white', 
            linewidth=0.5,
            alpha=1)
    

    plt.xlabel("X Position ($\mathbf{\mu m}$)", fontsize=30, fontweight='bold')
    plt.ylabel("Y Position ($\mathbf{\mu m}$)", fontsize=30, fontweight='bold')
    leg = plt.legend(
    loc='upper right', 
    title="Cell Types", 
    #fontsize=24, 
    title_fontsize=26, 
    markerscale=3,
    prop={'size': 26,'weight': 'bold'}  
)

    leg.get_title().set_fontweight('bold')
    plt.xticks(fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')
    

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().set_aspect('equal', adjustable='box') 
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(save_file, dpi=resolution, bbox_inches='tight')
    plt.show()


##############  Uniformly distributed ##############
def generate_scattered_fov(count_a=500, count_b=500, width=760, height=760, min_dist=5, seed=42):

    np.random.seed(seed)
    
    total_cells = count_a + count_b
    points = []
    
    rng = np.random.default_rng(seed)
    
    while len(points) < total_cells:
        candidate = rng.uniform(0, [width, height])
        
        if len(points) == 0:
            points.append(candidate)
        else:
            tree = KDTree(points)
            dist, _ = tree.query(candidate)
            
            if dist >= min_dist:
                points.append(candidate)
                
    points = np.array(points)
    
    labels = (['Type A'] * count_a) + (['Type B'] * count_b)
    rng.shuffle(labels)
    
    return pd.DataFrame({
        'X': points[:, 0],
        'Y': points[:, 1],
        'cell_type': labels
    })


############### Linear Perfect Separation #######################
def generate_perfect_segregation(count_a=500, count_b=500, width=1000, height=1000, min_dist=5, seed=42):
    rng = np.random.default_rng(seed)
    total_cells = count_a + count_b
    points = []
    
    while len(points) < total_cells:
        candidate = rng.uniform(0, [width, height])
        if len(points) == 0:
            points.append(candidate)
        else:

            tree = KDTree(points)
            dist, _ = tree.query(candidate)
            if dist >= min_dist:
                points.append(candidate)
                
    points = np.array(points)
    
    sorted_indices = np.argsort(points[:, 0])
    points = points[sorted_indices]
    
    labels = (['Type A'] * count_a) + (['Type B'] * count_b)
    
    return pd.DataFrame({
        'X': points[:, 0],
        'Y': points[:, 1],
        'cell_type': labels
    })




#####################  Clustered Segregation ##################


def generate_cluster_segregation(count_a=500, count_b=500, 
                          mode="a_clustered", # Options: "a_clustered", "b_clustered", "both_clustered"
                          num_seeds=5, cluster_mixing=30,
                          width=1000, height=1000, min_dist=5, seed=42):
    rng = np.random.default_rng(seed)
    total_cells = count_a + count_b
    
    points = []
    while len(points) < total_cells:
        candidate = rng.uniform(0, [width, height])
        if len(points) == 0:
            points.append(candidate)
        else:
            tree = KDTree(points)
            dist, _ = tree.query(candidate)
            if dist >= min_dist:
                points.append(candidate)
    points = np.array(points)

    labels = np.empty(total_cells, dtype=object)
    
    if mode == "a_clustered":
        seeds = rng.uniform(0, [width, height], size=(num_seeds, 2))
        dists = np.min(distance.cdist(points, seeds), axis=1)
        scores = dists + rng.normal(0, cluster_mixing, size=total_cells)
        idx = np.argsort(scores)
        labels[idx[:count_a]] = 'Type A'
        labels[idx[count_a:]] = 'Type B'

    elif mode == "b_clustered":
        seeds = rng.uniform(0, [width, height], size=(num_seeds, 2))
        dists = np.min(distance.cdist(points, seeds), axis=1)
        scores = dists + rng.normal(0, cluster_mixing, size=total_cells)
        idx = np.argsort(scores)
        labels[idx[:count_b]] = 'Type B'
        labels[idx[count_b:]] = 'Type A'

    elif mode == "both_clustered":
        seeds_a = rng.uniform(0, [width, height], size=(num_seeds, 2))
        seeds_b = rng.uniform(0, [width, height], size=(num_seeds, 2))
        
        dist_to_a = np.min(distance.cdist(points, seeds_a), axis=1)
        dist_to_b = np.min(distance.cdist(points, seeds_b), axis=1)
        
        scores = (dist_to_a - dist_to_b) + rng.normal(0, cluster_mixing, size=total_cells)
        idx = np.argsort(scores)
        labels[idx[:count_a]] = 'Type A'
        labels[idx[count_a:]] = 'Type B'

    return pd.DataFrame({'X': points[:, 0], 'Y': points[:, 1], 'cell_type': labels})




