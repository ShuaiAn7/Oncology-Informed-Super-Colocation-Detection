# -*- coding: utf-8 -*-
"""

@author: shuai
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
from pair_relation import count_neighbors_brute_force, count_neighbors_rtree, count_neighbors_space_partition, count_neighbors_sweepline
from pair_relation import generate_integer_adj_map, double_grid_mapping, super_block_mapping, varying_super_block



####################  Sensitivity Analysis #########################

fov_size = 760
nei_distance = 20
num_bins = int(np.ceil(fov_size / nei_distance))
run_repeat = 2  

complete_runtime = {}
for cell_num in range(1000, 5001, 1000):
    # Now, running this twice with seed 42 will give identical results
    #synth = generate_scattered_fov(count_a=cell_num, count_b=cell_num, width=fov_size, height=fov_size, min_dist=5, seed=42)
    #synth = generate_perfect_segregation(count_a=cell_num, count_b=cell_num, width=fov_size, height=fov_size, min_dist=5, seed=2026)
    #synth = generate_wavy_blurred_segregation(count_a=cell_num, count_b=cell_num, width=fov_size, height=fov_size, min_dist=5, blur_strength=40, seed=42)
    #synth = generate_cluster_segregation(count_a=cell_num, count_b=cell_num, mode="b_clustered", num_seeds=5, cluster_mixing=30, width=fov_size, height=fov_size, min_dist=5, seed=2026)
    #synth = generate_poisson_disc_fov(width=fov_size, height=fov_size, n_type_a=cell_num, n_type_b=cell_num, min_dist=5)
    
    pair_name = ('Type A', 'Type B')
    type_a = synth[synth['cell_type']=='Type A'].copy()
    type_b = synth[synth['cell_type']=='Type B'].copy()
  
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
    
    ##### double_grid_mapping ########
    adj_tensor, num_bins = generate_integer_adj_map(fov_length=fov_size, grid_length=nei_distance)
    neighbor_double_grid = double_grid_mapping(pair_name, type_a, type_b, distance=nei_distance, adj_tensor=adj_tensor, num_bins=num_bins)
    
    t = timeit.timeit(
    "double_grid_mapping(pair_name, type_a, type_b, distance=nei_distance, adj_tensor=adj_tensor, num_bins=num_bins)",
    globals=globals(), number=run_repeat)
    double_grid_avg_time_ms = (t / run_repeat) * 1000
    print(f"Average per run of double-grid-mapping: {(t/run_repeat)*1000:.3f} ms")
    
    ##### Super-Block-Mapping-2 without tensor mapping ##########
    neighbor_super_block_2 = super_block_mapping(pair_name, type_a, type_b, num_bins, distance=nei_distance)
    t = timeit.timeit("super_block_mapping(pair_name, type_a, type_b, num_bins, distance=nei_distance)", globals=globals(), number=run_repeat)
    super_block_2_avg_time_ms = (t / run_repeat) * 1000
    print(f"Average per run of super-block: {(t/run_repeat)*1000:.3f} ms")
    
 

  
########## Super-Block varying block size no mapping tensor #############
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


    
    time_list = [('Space Partition', space_partition_avg_time_ms),
             ('Sweepline', sweepline_avg_time_ms),
             ('Grid-Grids-Mapping', double_grid_avg_time_ms),
             ('Block-Grids-Mapping-2', super_block_2_avg_time_ms),
             ('Block-Grids-Mapping-Best', super_block_best_size_avg_time_ms)]
    
    complete_runtime[cell_num] = time_list






formatted_data = {size: dict(runtimes) for size, runtimes in complete_runtime.items()}
df_runtime = pd.DataFrame.from_dict(formatted_data, orient='index')


df_runtime = df_runtime.rename_axis('Cells per Type').reset_index()


line_styles = [
    '-o',  
    '--s', 
    '-.^', 
    ':d',  
    '-v',  
    '--*', 
    '-.p', 
    ':x',  
    '-h'   
]


ax = df_runtime.set_index('Cells per Type').plot(
    kind='line', 
    style=line_styles, 
    figsize=(8, 7),
    lw=2.5,
    ms=12)


plt.xlabel('Number of Cells per Type', fontsize=26, fontweight='bold')
plt.ylabel('Runtime (milliseconds)', fontsize=26, fontweight='bold')

ax.tick_params(axis='both', labelsize=22)

for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

plt.grid(True, linestyle='--', alpha=0.7)
leg = plt.legend(
    labels = ['Space Partition', 'Sweepline', 'GGM', 'BGM-2', 'BGM-Best'],
    loc='best', 
    title="Method:", 
    #fontsize=24, 
    title_fontsize=23, 
    markerscale=1.2,
    prop={'size': 23,'weight': 'bold'}  
)
leg.get_title().set_fontweight('bold')


plt.tight_layout()
plt.savefig('', dpi=400, bbox_inches='tight')
plt.show() 

