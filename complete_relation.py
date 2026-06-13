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
from scipy.spatial.distance import jensenshannon
from itertools import combinations
import matplotlib as mpl  
from itertools import permutations
import time
import math
import random
from rtree import index
import numpy as np
from collections import defaultdict
from pair_relation import count_neighbors_brute_force, count_neighbors_rtree, count_neighbors_space_partition, super_block_mapping




################################ Brute force approach by looping cells  ##################


def count_neighbors_brute_force_no_self_type(
    fov: pd.DataFrame,
    R: float,
    x_col: str = "X",
    y_col: str = "Y",
    cell_type_col: str = "cell_type"
):

    out = fov.copy()
    coords = out[[x_col, y_col]].to_numpy(float).tolist()
    N = len(out)
    
    types = pd.Categorical(out[cell_type_col])
    type_names = list(types.categories)
    codes = np.asarray(types.codes).tolist() 
    K = len(type_names)

    counts_out = np.zeros((N, K), dtype=np.int32)
    R_sq = R**2

    for i in range(N):
        current_anchor_counts = [0] * K
        xi, yi = coords[i]
        anchor_type_code = codes[i]
        
        for j in range(N):
            if i == j:
                continue
                
            xj, yj = coords[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            
            if dist_sq <= R_sq:
                neighbor_type_code = codes[j]
                current_anchor_counts[neighbor_type_code] += 1

        current_anchor_counts[anchor_type_code] = 0 
        
        counts_out[i, :] = current_anchor_counts

    for i, t_name in enumerate(type_names):
        out[f"Num of {t_name} neighbors"] = counts_out[:, i]

    return out



########################  Generate Neighbor-Relation Proposed #####################


def get_master_map(fov_width, fov_height, step_inner, step_outer, distance):
    master_map = {}

    nx = int(np.ceil(fov_width / step_inner))
    ny = int(np.ceil(fov_height / step_inner))
    
    for ix in range(nx):
        for iy in range(ny):
            ikey = ix * 1000000 + iy
            
            x0, x1 = ix * step_inner, (ix + 1) * step_inner
            y0, y1 = iy * step_inner, (iy + 1) * step_inner
            
            search_x_min = x0 - distance
            search_x_max = x1 + distance
            search_y_min = y0 - distance
            search_y_max = y1 + distance
            
            ox_s = int(np.floor(search_x_min / step_outer))
            ox_e = int(np.floor(search_x_max / step_outer))
            oy_s = int(np.floor(search_y_min / step_outer))
            oy_e = int(np.floor(search_y_max / step_outer))
            
            keys = []
            for ox in range(ox_s, ox_e + 1):
                for oy in range(oy_s, oy_e + 1):
                    keys.append(ox * 1000000 + oy)
            
            master_map[ikey] = keys
            
    return master_map





def generate_counts_from_R(fov, relation_R):

    id_to_idx = {cid: i for i, cid in enumerate(fov['cell_id'])}
    id_to_type = dict(zip(fov['cell_id'], fov['cell_type']))
    
    type_names = sorted(fov['cell_type'].unique())
    type_to_col_idx = {t: i for i, t in enumerate(type_names)}
    
    N = len(fov)
    K = len(type_names)
    
    counts_matrix = np.zeros((N, K), dtype=np.int32)
    
    for id_i, id_j in relation_R:
        idx_i = id_to_idx[id_i]
        idx_j = id_to_idx[id_j]
        
        type_i = id_to_type[id_i]
        type_j = id_to_type[id_j]
        
        counts_matrix[idx_i, type_to_col_idx[type_j]] += 1
        
        counts_matrix[idx_j, type_to_col_idx[type_i]] += 1

    output_fov = fov.copy()
    for i, t_name in enumerate(type_names):
        col_name = f"Num of {t_name} neighbors"
        output_fov[col_name] = counts_matrix[:, i]
        
    return output_fov




def generate_relation_R_hierarchical(fov, R_dist, master_map):
    ids = fov['cell_id'].tolist()
    xs = fov['X'].tolist()
    ys = fov['Y'].tolist()
    type_cats = fov['cell_type'].astype('category').cat
    cts = type_cats.codes.tolist()
    
    N = len(fov)
    relation_R = set()
    R_sq = R_dist**2
    step_inner = (math.sqrt(2) / 2.0) * R_dist

    grid_inner = {}
    grid_outer = {}
    
    for i in range(N):
        ikey = int(xs[i] // step_inner) * 1000000 + int(ys[i] // step_inner)
        okey = int(xs[i] // R_dist) * 1000000 + int(ys[i] // R_dist)
        
        t_code = cts[i]
        
        if ikey not in grid_inner: grid_inner[ikey] = {}
        if t_code not in grid_inner[ikey]: grid_inner[ikey][t_code] = []
        grid_inner[ikey][t_code].append(i)
        
        if okey not in grid_outer: grid_outer[okey] = []
        grid_outer[okey].append(i)

    for ikey, type_map in grid_inner.items():
        type_codes_in_bin = list(type_map.keys())
        
        for i_t_idx in range(len(type_codes_in_bin)):
            t_i = type_codes_in_bin[i_t_idx]
            indices_i = type_map[t_i]
            
            for j_t_idx in range(i_t_idx + 1, len(type_codes_in_bin)):
                t_j = type_codes_in_bin[j_t_idx]
                indices_j = type_map[t_j]
                
                for idx_i in indices_i:
                    id_i = ids[idx_i]
                    for idx_j in indices_j:
                        id_j = ids[idx_j]
                        if id_i < id_j: relation_R.add((id_i, id_j))
                        else: relation_R.add((id_j, id_i))

        potential_outer_keys = master_map.get(ikey, [])
        inner_indices_all = [idx for lst in type_map.values() for idx in lst]
        inner_set = set(inner_indices_all)
        
        candidates = []
        for okey in potential_outer_keys:
            if okey in grid_outer:
                for c_idx in grid_outer[okey]:
                    if c_idx not in inner_set:
                        candidates.append(c_idx)
        
        for t_i, indices_i in type_map.items():
            for idx_i in indices_i:
                ax, ay, id_i = xs[idx_i], ys[idx_i], ids[idx_i]
                
                for cand_idx in candidates:
                    if t_i == cts[cand_idx]:
                        continue
                        
                    dx = ax - xs[cand_idx]
                    dy = ay - ys[cand_idx]
                    if dx*dx + dy*dy <= R_sq:
                        id_j = ids[cand_idx]
                        if id_i < id_j: relation_R.add((id_i, id_j))
                        else: relation_R.add((id_j, id_i))

    return relation_R




################### Use R tree to generate relation R ##############


def generate_relation_R_rtree(
    fov, 
    distance=20.0, 
    x_col: str = "X", 
    y_col: str = "Y"
):

    ids = fov['cell_id'].values
    types = fov['cell_type'].values
    coords = fov[[x_col, y_col]].to_numpy()
    N = len(fov)
    
    relation_R = set()
    dist_sq_threshold = distance**2

    p = index.Property()
    idx = index.Index(properties=p)
    for i in range(N):
        x, y = coords[i]
        idx.insert(i, (x, y, x, y))

    for i in range(N):
        ax, ay = coords[i]
        type_i = types[i]
        id_i = ids[i]
        
        search_box = (ax - distance, ay - distance, ax + distance, ay + distance)
        
        candidates = list(idx.intersection(search_box))
        
        for cand_idx in candidates:

            if i < cand_idx and type_i != types[cand_idx]:
                
                cx, cy = coords[cand_idx]
                dx = ax - cx
                dy = ay - cy
                if (dx*dx + dy*dy) <= dist_sq_threshold:
                    id_j = ids[cand_idx]
                    relation_R.add(tuple(sorted((id_i, id_j))))

    return relation_R



########  Use space partition to generate relation R ##################


def generate_relation_R_space_partition(
    fov, 
    distance: float = 20.0, 
    x_col: str = "X", 
    y_col: str = "Y"
):


    ids = fov['cell_id'].values
    types = fov['cell_type'].values
    coords = fov[[x_col, y_col]].to_numpy()
    N = len(fov)
    
    relation_R = set()
    dist_sq_threshold = distance ** 2

    grid = defaultdict(list)
    for i in range(N):
        ix, iy = int(coords[i, 0] // distance), int(coords[i, 1] // distance)
        grid[(ix, iy)].append(i)

    for i in range(N):
        ax, ay = coords[i]
        type_i = types[i]
        id_i = ids[i]
        
        ix_c, iy_c = int(ax // distance), int(ay // distance)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                bin_key = (ix_c + dx, iy_c + dy)
                
                if bin_key in grid:
                    for cand_idx in grid[bin_key]:

                        if i < cand_idx and type_i != types[cand_idx]:
                            
                            cx, cy = coords[cand_idx]
                            dx_val = ax - cx
                            dy_val = ay - cy                     
                            if (dx_val**2 + dy_val**2) <= dist_sq_threshold:
                                id_j = ids[cand_idx]
                                relation_R.add(tuple(sorted((id_i, id_j))))

    return relation_R



#################  sweep plane ###################
def generate_relation_R_sweepline(fov, distance=20.0):

    sorted_df = fov.sort_values(by='X').copy()
    ids = sorted_df['cell_id'].values
    xs = sorted_df['X'].values
    ys = sorted_df['Y'].values
    cts = sorted_df['cell_type'].values
    N = len(sorted_df)
    
    relation_R = set()
    dist_sq_threshold = distance**2
    
    for i in range(N):
        xi, yi, type_i, id_i = xs[i], ys[i], cts[i], ids[i]
        
        for j in range(i + 1, N):

            dx = xs[j] - xi
            if dx > distance:
                break

            if type_i != cts[j]:

                dy = ys[j] - yi
                if abs(dy) <= distance:

                    if (dx*dx + dy*dy) <= dist_sq_threshold:
                        id_j = ids[j]
                        relation_R.add(tuple(sorted((id_i, id_j))))
                        
    return relation_R

