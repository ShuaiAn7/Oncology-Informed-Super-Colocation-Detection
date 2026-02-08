# -*- coding: utf-8 -*-
"""

@author: an000033
"""

import os
import pyreadr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from scipy.spatial.distance import jensenshannon
from itertools import combinations
import matplotlib as mpl  
from itertools import permutations
import time
import timeit
from collections import defaultdict
from scipy.stats import qmc
from rtree import index
from synthetic_data import generate_scattered_fov, generate_perfect_segregation, generate_cluster_segregation





###################### Brute Force ################

def count_neighbors_brute_force(
    pair_name,
    anchor_df,
    surround_df,
    distance: float = 20.0,
    x_col: str = "X",
    y_col: str = "Y",
):


    d2 = distance ** 2
    counts = []

    for _, a_row in anchor_df.iterrows():
        ax = float(a_row[x_col])
        ay = float(a_row[y_col])

        c = 0

        for _, s_row in surround_df.iterrows():
            bx = float(s_row[x_col])
            by = float(s_row[y_col])

            dx = bx - ax
            dy = by - ay

            if dx*dx + dy*dy <= d2:
                c += 1

        counts.append(c)

    out = anchor_df.copy()
    out[f"Num of {pair_name[1]} neighbors"] = counts
    return out



#######################   R tree  ############################


def count_neighbors_rtree(
    pair_name,
    anchor_df,
    surround_df,
    distance = 20.0,
    x_col: str = "X",
    y_col: str = "Y"):
  
    anchor_type, neighbor_type = pair_name
    col_name = f"Num of {neighbor_type} neighbors"

    if anchor_df.empty:
        return anchor_df
    if surround_df.empty:
        anchor_df[col_name] = 0
        return anchor_df


    p = index.Property()
    idx = index.Index(properties=p)

    neighbor_coords = surround_df[[x_col, y_col]].to_numpy()
    for i, (nx, ny) in enumerate(neighbor_coords):
        idx.insert(i, (nx, ny, nx, ny))

    anchor_coords = anchor_df[[x_col, y_col]].to_numpy()
    counts = []
    
    for ax, ay in anchor_coords:

        search_box = (ax - distance, ay - distance, ax + distance, ay + distance)
        
        candidates = list(idx.intersection(search_box))
        
        real_neighbors = 0
        if candidates:
            candidate_coords = neighbor_coords[candidates]
            dist_sq = np.sum((candidate_coords - np.array([ax, ay]))**2, axis=1)
            real_neighbors = np.sum(dist_sq <= distance**2)
            
        counts.append(real_neighbors)

    anchor_df[col_name] = counts
    return anchor_df



##########  Space Partition ####################################################


def count_neighbors_space_partition(
    pair_name,
    anchor_df,
    surround_df,
    distance: float = 20.0,
    x_col: str = "X",
    y_col: str = "Y"):
    
    anchor_type, neighbor_type = pair_name
    col_name = f"Num of {neighbor_type} neighbors"

    if anchor_df.empty:
        return anchor_df
    if surround_df.empty:
        anchor_df[col_name] = 0
        return anchor_df

    grid = defaultdict(list)
    neighbor_coords = surround_df[[x_col, y_col]].to_numpy()
    
    for i in range(len(neighbor_coords)):
        nx, ny = neighbor_coords[i]
        ix, iy = int(nx // distance), int(ny // distance)
        grid[(ix, iy)].append(neighbor_coords[i])

    for key in grid:
        grid[key] = np.array(grid[key])

    anchor_coords = anchor_df[[x_col, y_col]].to_numpy()
    counts = np.zeros(len(anchor_df), dtype=int)
    dist_sq_threshold = distance ** 2

    for i, (ax, ay) in enumerate(anchor_coords):
        ix_center, iy_center = int(ax // distance), int(ay // distance)
        
        candidates = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                bin_key = (ix_center + dx, iy_center + dy)
                if bin_key in grid:
                    candidates.append(grid[bin_key])
        
        if not candidates:
            counts[i] = 0
            continue

        candidate_pool = np.vstack(candidates)
        
        diffs = candidate_pool - np.array([ax, ay])
        sq_dists = np.sum(diffs**2, axis=1)
        
        counts[i] = np.sum(sq_dists <= dist_sq_threshold)

    anchor_df[col_name] = counts
    return anchor_df



#######################  Sweepline approach #########################


def count_neighbors_sweepline(
    pair_name,
    anchor_df,
    surround_df,
    distance: float = 20.0,
    x_col: str = "X",
    y_col: str = "Y"
):
    anchor_type, neighbor_type = pair_name
    col_name = f"Num of {neighbor_type} neighbors"

    if anchor_df.empty:
        return anchor_df
    if surround_df.empty:
        anchor_df[col_name] = 0
        return anchor_df

    surround_sorted = surround_df.sort_values(by=x_col)
    neighbor_coords = surround_sorted[[x_col, y_col]].to_numpy()
    neighbor_x = neighbor_coords[:, 0]
    
    anchor_coords = anchor_df[[x_col, y_col]].to_numpy()
    counts = np.zeros(len(anchor_df), dtype=int)
    dist_sq_threshold = distance ** 2

    for i, (ax, ay) in enumerate(anchor_coords):

        left_idx = np.searchsorted(neighbor_x, ax - distance, side='left')
        right_idx = np.searchsorted(neighbor_x, ax + distance, side='right')
        
        if left_idx == right_idx:
            counts[i] = 0
            continue

        candidate_pool = neighbor_coords[left_idx:right_idx]
        
        dy_mask = np.abs(candidate_pool[:, 1] - ay) <= distance
        valid_candidates = candidate_pool[dy_mask]
        
        if len(valid_candidates) == 0:
            counts[i] = 0
            continue

        diffs = valid_candidates - np.array([ax, ay])
        sq_dists = np.sum(diffs**2, axis=1)
        counts[i] = np.sum(sq_dists <= dist_sq_threshold)

    anchor_df[col_name] = counts
    return anchor_df


########################## Proposed Methods #########################
############ GGM  ##############

def generate_integer_adj_map(fov_length, grid_length):
    num_bins = int(np.ceil(fov_length / grid_length))

    adj_tensor = np.full((num_bins, num_bins, 9, 2), -1, dtype=int)
    
    for i in range(num_bins):
        for j in range(num_bins):
            idx = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < num_bins and 0 <= ny < num_bins:
                        adj_tensor[i, j, idx] = [nx, ny]
                    idx += 1
    return adj_tensor, num_bins


def double_grid_mapping(pair_name, anchor_df, surround_df, adj_tensor, num_bins, distance=20.0, x_col="X", y_col="Y"
):
    anchor_type, neighbor_type = pair_name
    col_name = f"Num of {neighbor_type} neighbors"
     
    grid_surround = [[[] for _ in range(num_bins)] for _ in range(num_bins)]
    grid_anchor = [[{'coords': [], 'idx': []} for _ in range(num_bins)] for _ in range(num_bins)]
    
    s_coords = surround_df[[x_col, y_col]].to_numpy()
    for i in range(len(s_coords)):
        ix, iy = int(s_coords[i,0] // distance), int(s_coords[i,1] // distance)
        if 0 <= ix < num_bins and 0 <= iy < num_bins:
            grid_surround[ix][iy].append(s_coords[i])

    a_coords_raw = anchor_df[[x_col, y_col]].to_numpy()
    for i in range(len(a_coords_raw)):
        ix, iy = int(a_coords_raw[i,0] // distance), int(a_coords_raw[i,1] // distance)
        if 0 <= ix < num_bins and 0 <= iy < num_bins:
            grid_anchor[ix][iy]['coords'].append(a_coords_raw[i])
            grid_anchor[ix][iy]['idx'].append(i)

    counts = np.zeros(len(anchor_df), dtype=int)
    dist_sq = distance ** 2

    for i in range(num_bins):
        for j in range(num_bins):
            cell_data = grid_anchor[i][j]
            if not cell_data['coords']: continue
            

            neighbor_indices = adj_tensor[i, j] 
            
            candidates = []
            for ni, nj in neighbor_indices:
                if ni != -1: 
                    candidates.extend(grid_surround[ni][nj])
            
            if not candidates: continue
            
            curr_a_coords = np.array(cell_data['coords'])
            curr_s_pool = np.array(candidates)
            
            diffs = curr_a_coords[:, np.newaxis, :] - curr_s_pool[np.newaxis, :, :]
            sq_dists = np.sum(diffs**2, axis=2)
            counts[cell_data['idx']] = np.sum(sq_dists <= dist_sq, axis=1)

    anchor_df[col_name] = counts
    return anchor_df




####################### BGM #####################################


def super_block_mapping(pair_name, anchor_df, surround_df, 
    num_bins, 
    distance=20.0, 
    x_col="X", 
    y_col="Y"
):
    anchor_type, neighbor_type = pair_name
    col_name = f"Num of {neighbor_type} neighbors"
  
    
    grid_surround = [[[] for _ in range(num_bins)] for _ in range(num_bins)]
    grid_anchor = [[{'coords': [], 'idx': []} for _ in range(num_bins)] for _ in range(num_bins)]
    
    s_coords = surround_df[[x_col, y_col]].to_numpy()
    for i in range(len(s_coords)):
        ix, iy = int(s_coords[i,0] // distance), int(s_coords[i,1] // distance)
        if 0 <= ix < num_bins and 0 <= iy < num_bins:
            grid_surround[ix][iy].append(s_coords[i])

    a_coords_raw = anchor_df[[x_col, y_col]].to_numpy()
    for i in range(len(a_coords_raw)):
        ix, iy = int(a_coords_raw[i,0] // distance), int(a_coords_raw[i,1] // distance)
        if 0 <= ix < num_bins and 0 <= iy < num_bins:
            grid_anchor[ix][iy]['coords'].append(a_coords_raw[i])
            grid_anchor[ix][iy]['idx'].append(i)

    counts = np.zeros(len(anchor_df), dtype=int)
    dist_sq = distance ** 2

    for i in range(0, num_bins, 2):
        for j in range(0, num_bins, 2):
            
            block_a_coords = []
            block_a_indices = []
            
            for bi in [i, i+1]:
                for bj in [j, j+1]:
                    if bi < num_bins and bj < num_bins:
                        data = grid_anchor[bi][bj]
                        if data['coords']:
                            block_a_coords.append(np.array(data['coords']))
                            block_a_indices.append(np.array(data['idx']))
            
            if not block_a_coords:
                continue
                
            curr_a_coords = np.vstack(block_a_coords)
            curr_a_indices = np.concatenate(block_a_indices)

            candidate_list = []
            for ni in range(i - 1, i + 3):
                for nj in range(j - 1, j + 3):
                    if 0 <= ni < num_bins and 0 <= nj < num_bins:
                        if grid_surround[ni][nj]:
                            candidate_list.extend(grid_surround[ni][nj])
            
            if not candidate_list:
                continue

            curr_s_pool = np.array(candidate_list)
            
            diffs = curr_a_coords[:, np.newaxis, :] - curr_s_pool[np.newaxis, :, :]
            sq_dists = np.sum(diffs**2, axis=2)
            
            counts[curr_a_indices] = np.sum(sq_dists <= dist_sq, axis=1)

    anchor_df[col_name] = counts
    return anchor_df





###############################  BGM-Best ###########################

def varying_super_block(
    pair_name,
    anchor_df,
    surround_df,
    num_bins, 
    block_size=2, 
    distance=20.0, 
    x_col="X", 
    y_col="Y"
):
    anchor_type, neighbor_type = pair_name
    col_name = f"Num of {neighbor_type} neighbors"
    
    grid_surround = [[[] for _ in range(num_bins)] for _ in range(num_bins)]
    grid_anchor = [[{'coords': [], 'idx': []} for _ in range(num_bins)] for _ in range(num_bins)]
    
    for df, grid, is_anchor in [(surround_df, grid_surround, False), (anchor_df, grid_anchor, True)]:
        coords = df[[x_col, y_col]].to_numpy()
        for i in range(len(coords)):
            ix, iy = int(coords[i,0] // distance), int(coords[i,1] // distance)
            if 0 <= ix < num_bins and 0 <= iy < num_bins:
                if is_anchor:
                    grid[ix][iy]['coords'].append(coords[i])
                    grid[ix][iy]['idx'].append(i)
                else:
                    grid[ix][iy].append(coords[i])

    counts = np.zeros(len(anchor_df), dtype=int)
    dist_sq = distance ** 2

    for i in range(0, num_bins, block_size):
        for j in range(0, num_bins, block_size):
            
            block_a_coords = []
            block_a_indices = []
            for bi in range(i, min(i + block_size, num_bins)):
                for bj in range(j, min(j + block_size, num_bins)):
                    data = grid_anchor[bi][bj]
                    if data['coords']:
                        block_a_coords.append(np.array(data['coords']))
                        block_a_indices.append(np.array(data['idx']))
            
            if not block_a_coords:
                continue
                
            curr_a_coords = np.vstack(block_a_coords)
            curr_a_indices = np.concatenate(block_a_indices)

            candidate_list = []
            
            x_start, x_end = max(0, i - 1), min(num_bins, i + block_size + 1)
            y_start, y_end = max(0, j - 1), min(num_bins, j + block_size + 1)
            
            for ni in range(x_start, x_end):
                for nj in range(y_start, y_end):
                    if grid_surround[ni][nj]:
                        candidate_list.extend(grid_surround[ni][nj])
            
            if not candidate_list:
                continue

            curr_s_pool = np.array(candidate_list)
            diffs = curr_a_coords[:, np.newaxis, :] - curr_s_pool[np.newaxis, :, :]
            sq_dists = np.sum(diffs**2, axis=2)
            counts[curr_a_indices] = np.sum(sq_dists <= dist_sq, axis=1)

    anchor_df[col_name] = counts
    return anchor_df
  

  



if __name__ == "__main__":
    ###################  Analysis #########################
    
    fov_size = 760
    nei_distance = 20
    num_bins = int(np.ceil(fov_size / nei_distance))
    run_repeat = 2  
    complete_runtime = {}
    for cell_num in range(1000, 5001, 1000):
        # Now, running this twice with seed 42 will give identical results
        synth = generate_cluster_segregation(count_a=cell_num, count_b=cell_num, mode="a_clustered", num_seeds=5, cluster_mixing=30, width=fov_size, height=fov_size, min_dist=5, seed=2026)
        #synth = generate_poisson_disc_fov(width=fov_size, height=fov_size, n_type_a=cell_num, n_type_b=cell_num, min_dist=5)
        #synth = generate_scattered_fov(count_a=cell_num, count_b=cell_num, width=fov_size, height=fov_size, min_dist=5, seed=42)
        #synth = generate_perfect_segregation(count_a=cell_num, count_b=cell_num, width=fov_size, height=fov_size, min_dist=5, seed=42)
        #synth = generate_wavy_blurred_segregation(count_a=cell_num, count_b=cell_num, width=fov_size, height=fov_size, min_dist=5, blur_strength=40, seed=42)
        
        pair_name = ('Type_A', 'Type_B')
        type_a = synth[synth['cell_type']=='Type_A'].copy()
        type_b = synth[synth['cell_type']=='Type_B'].copy()
      
        ##### Brute force approach ###########
        neighbor_brute_force = count_neighbors_brute_force(pair_name, type_a, type_b, distance=nei_distance)
        t = timeit.timeit("count_neighbors_brute_force(pair_name, type_a, type_b, distance=nei_distance)", globals=globals(), number=1)
        basic_approach_avg_time_ms = (t / 100) * 1000
        print(f"Average per run of the brute force approach: {(t/100)*1000:.3f} ms")
        
        
        ###### R Tree ############
        neighbor_rtree = count_neighbors_rtree(pair_name, type_a, type_b, distance=nei_distance)
        t = timeit.timeit("count_neighbors_rtree(pair_name, type_a, type_b, distance=nei_distance)", globals=globals(), number=run_repeat)
        rtree_avg_time_ms = (t / run_repeat) * 1000
        print(f"Average per run of R tree: {(t/run_repeat)*1000:.3f} ms")
        
        ##### Space Partition ########
        neighbor_space_partition = count_neighbors_space_partition(pair_name, type_a, type_b, distance=nei_distance)
        t = timeit.timeit("count_neighbors_space_partition(pair_name, type_a, type_b, distance=nei_distance)", globals=globals(), number=run_repeat)
        space_partition_avg_time_ms = (t / run_repeat) * 1000
        print(f"Average per run of space partition: {(t/run_repeat)*1000:.3f} ms")
        
        
        ##### Sweepline ########
        neighbor_sweepline = count_neighbors_sweepline(pair_name, type_a, type_b, distance=nei_distance)
        t = timeit.timeit("count_neighbors_sweepline(pair_name, type_a, type_b, distance=nei_distance)", globals=globals(), number=run_repeat)
        sweepline_avg_time_ms = (t / run_repeat) * 1000
        print(f"Average per run of sweepline: {(t/run_repeat)*1000:.3f} ms")
        
        ##### GGM ########
        adj_tensor, num_bins = generate_integer_adj_map(fov_length=fov_size, grid_length=nei_distance)
        neighbor_double_grid = double_grid_mapping(pair_name, type_a, type_b, distance=nei_distance, adj_tensor=adj_tensor, num_bins=num_bins)
        
        t = timeit.timeit(
        "double_grid_mapping(pair_name, type_a, type_b, distance=nei_distance, adj_tensor=adj_tensor, num_bins=num_bins)",
        globals=globals(), number=run_repeat)
        double_grid_avg_time_ms = (t / run_repeat) * 1000
        print(f"Average per run of double-grid-mapping: {(t/run_repeat)*1000:.3f} ms")
        
        ##### BGM ##########
        neighbor_super_block_2 = super_block_mapping(pair_name, type_a, type_b, num_bins, distance=nei_distance)
        t = timeit.timeit("super_block_mapping(pair_name, type_a, type_b, num_bins, distance=nei_distance)", globals=globals(), number=run_repeat)
        super_block_2_avg_time_ms = (t / run_repeat) * 1000
        print(f"Average per run of super-block: {(t/run_repeat)*1000:.3f} ms")
        

    
      
    ########## BGM-Best #############
        results = []  
        for i in range(1, 10):   
            t = timeit.timeit(
                f"varying_super_block(pair_name, type_a, type_b, num_bins, block_size={i}, distance=nei_distance)", 
                globals=globals(), 
                number=run_repeat)
            avg_ms = (t / run_repeat) * 1000
            results.append((i, avg_ms)) 

        
        best_size, best_time = min(results, key=lambda x: x[1])
        
        neighbor_super_block_best_size = varying_super_block(pair_name, type_a, type_b, num_bins,
                                                                  block_size=best_size, distance=nei_distance)
        t = timeit.timeit(
        "varying_super_block(pair_name, type_a, type_b, num_bins, block_size=best_size, distance=nei_distance)",
        globals=globals(), number=run_repeat)
        super_block_best_size_avg_time_ms = (t / run_repeat) * 1000
        print(f"Average per run of super-block with the best size: {(t/run_repeat)*1000:.3f} ms")
    

        
        time_list = [('Rtree', rtree_avg_time_ms),
                 ('Space partition', space_partition_avg_time_ms),
                 ('Sweepline', sweepline_avg_time_ms),
                 ('Grid-Grids-Mapping', double_grid_avg_time_ms),
                 ('Block-Grids-Mapping-2', super_block_2_avg_time_ms),
                 ('Block-Grids-Mapping-Best', super_block_best_size_avg_time_ms)]
        
        complete_runtime[cell_num] = time_list
    


    formatted_data = {size: dict(runtimes) for size, runtimes in complete_runtime.items()}
    df_runtime = pd.DataFrame.from_dict(formatted_data, orient='index')

    df_runtime = df_runtime.rename_axis('Cells per Type').reset_index()
    df_runtime = df_runtime.drop(['Rtree'], axis=1)
    
    line_styles = [
        '-o',  
        '--s', 
        '-.^', 
        ':d',  
        '-v',  
        '--*', 
        '-.p', 
        ':x',  
        '-h'   ]
    
 
    ax = df_runtime.set_index('Cells per Type').plot(
        kind='line', 
        style=line_styles, 
        figsize=(10, 6)
    )
    
    plt.title('Runtime by Number of Cells', fontsize=14)
    plt.xlabel('Number of Cells per Type')
    plt.ylabel('Milliseconds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Method:", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()    



