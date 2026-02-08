# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 09:32:12 2026

@author: shuai
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
from collections import defaultdict
from rtree import index
import numpy as np
from pair_relation import count_neighbors_brute_force, count_neighbors_rtree, count_neighbors_space_partition, super_block_mapping
from complete_relation import get_master_map
from complete_relation import generate_counts_from_R, generate_relation_R_hierarchical, generate_relation_R_rtree
from complete_relation import generate_relation_R_space_partition, generate_relation_R_sweepline





cells = pyreadr.read_r("")[None]

fov_summary = cells.groupby(['imageID', 'cell_type']).size().unstack(fill_value=0)
fov_summary['Total Cells'] = fov_summary.sum(axis=1)
fov_summary['Total Cells'].describe()
sorted_image_ids = fov_summary.sort_values('Total Cells', ascending=True).index.tolist()


n = 55  # The size of each quarter
list_of_lists = [sorted_image_ids[i:i + n] for i in range(0, len(sorted_image_ids), n)]
df_quarters = [
    [cells[cells['imageID'] == image_id] for image_id in id_bin]
    for id_bin in list_of_lists]


id_cols = ['imageID', 'cell_id', 'cell_type']



runtime_FOV_density = {"Rtree": {},
                   "Sweepline": {},
                   "Space Partition": {},
                   "Proposed": {}}





########################  Generate Neighbor-Relation Proposed #####################


nei_distance = 20
s_inner = (np.sqrt(2) / 2.0) * nei_distance
s_outer = nei_distance
fov_width = 760
fov_height = 760

m_map = get_master_map(fov_width, fov_height, s_inner, s_outer, nei_distance)

for i in range(4):
    lst = df_quarters[i]
    start_time = time.time()
    for fov in lst:
        relation_R = generate_relation_R_hierarchical(fov, nei_distance, m_map)
        neighbor_all_relationR = generate_counts_from_R(fov, relation_R)
        count_cols = [c for c in neighbor_all_relationR.columns if "Num of" in c]
        final_columns = id_cols + count_cols
        neighbor_all_relationR = neighbor_all_relationR[final_columns]
        neighbor_all_relationR = neighbor_all_relationR.sort_values(['imageID', 'cell_id']).reset_index(drop=True)
        neighbor_all_relationR[count_cols] = neighbor_all_relationR[count_cols].astype('int64')
    end_time = time.time()
    runtime_proposed = end_time - start_time
    runtime_FOV_density['Proposed'][f"Q{i+1}"] = runtime_proposed
    print(f'Total runtime of inner-outer grids method of quarter {i} is {end_time - start_time} seconds')



################### Use R tree to generate relation R ##############


for i in range(4):
    lst = df_quarters[i]
    start_time = time.time()
    for fov in lst:
        relation_R_rtree = generate_relation_R_rtree(fov)
        neighbor_all_relationR_rtree = generate_counts_from_R(fov, relation_R_rtree)
        count_cols = [c for c in neighbor_all_relationR_rtree.columns if "Num of" in c]
        final_columns = id_cols + count_cols
        neighbor_all_relationR_rtree = neighbor_all_relationR_rtree[final_columns]
        neighbor_all_relationR_rtree = neighbor_all_relationR_rtree.sort_values(['imageID', 'cell_id']).reset_index(drop=True)
        neighbor_all_relationR_rtree[count_cols] = neighbor_all_relationR_rtree[count_cols].astype('int64')
    end_time = time.time()
    runtime_rtree = end_time - start_time
    runtime_FOV_density['Rtree'][f"Q{i+1}"] = runtime_rtree
    print(f'Total runtime of R tree is {end_time - start_time} seconds')




########  Use space partition to generate relation R ##################


for i in range(4):
    lst = df_quarters[i]
    start_time = time.time()
    for fov in lst:
        relation_R_sp = generate_relation_R_space_partition(fov, distance=20)
        neighbor_all_relationR_sp = generate_counts_from_R(fov, relation_R_sp)
        count_cols = [c for c in neighbor_all_relationR_sp.columns if "Num of" in c]
        final_columns = id_cols + count_cols
        neighbor_all_relationR_sp = neighbor_all_relationR_sp[final_columns]
        neighbor_all_relationR_sp = neighbor_all_relationR_sp.sort_values(['imageID', 'cell_id']).reset_index(drop=True)
        neighbor_all_relationR_sp[count_cols] = neighbor_all_relationR_sp[count_cols].astype('int64')
    end_time = time.time()
    runtime_space_partition = end_time - start_time
    runtime_FOV_density['Space Partition'][f"Q{i+1}"] = runtime_space_partition
    print(f'Total runtime of space partition is {end_time - start_time} seconds')





#################  sweep plane ###################

for i in range(4):
    lst = df_quarters[i]
    start_time = time.time()
    for fov in lst:
        relation_R_sweepline = generate_relation_R_sweepline(fov)
        neighbor_all_relationR_sweepline = generate_counts_from_R(fov, relation_R_sweepline)
        count_cols = [c for c in neighbor_all_relationR_sweepline.columns if "Num of" in c]
        final_columns = id_cols + count_cols
        neighbor_all_relationR_sweepline = neighbor_all_relationR_sweepline[final_columns]
        neighbor_all_relationR_sweepline = neighbor_all_relationR_sweepline.sort_values(['imageID', 'cell_id']).reset_index(drop=True)
        neighbor_all_relationR_sweepline[count_cols] = neighbor_all_relationR_sweepline[count_cols].astype('int64')
    end_time = time.time()
    runtime_sweepline = end_time - start_time
    runtime_FOV_density['Sweepline'][f"Q{i+1}"] = runtime_sweepline
    print(f'Total runtime of sweep plane is {end_time - start_time} seconds')




df = pd.DataFrame(runtime_FOV_density)


plt.figure(figsize=(10, 6))

plt.plot(df.index, df['Rtree'],       marker='s', markersize=6, linestyle='--', label='Rtree')
plt.plot(df.index, df['Space Partition'], marker='^', markersize=6, linestyle='-.', label='Space Partition')
plt.plot(df.index, df['Sweepline'],   marker='d', markersize=6, linestyle=':',  label='Sweepline')

plt.plot(df.index, df['Proposed'],    marker='*', linestyle='-',  label='Proposed Method', 
         linewidth=1, color='red', markersize=8)

plt.xlabel('Number of Cells per FOV', fontsize=12)
plt.ylabel('Runtime (seconds)', fontsize=12)
plt.legend(title="Methods")
plt.grid(True, linestyle='--', alpha=0.5)



import pandas as pd
import matplotlib.pyplot as plt

colors = ['blue', '#FFD700', 'green', 'red'] 

ax = df.plot(kind='bar', 
             figsize=(10, 6), 
             color=colors, 
             width=0.8, 
             edgecolor='black')

ax.tick_params(axis='both', labelsize=20)

plt.xlabel('Quarters of Data by Density', fontsize=22)
plt.ylabel('Runtime (seconds)', fontsize=22)
plt.legend(title="Methods", fontsize=21, title_fontsize=21)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.xticks(rotation=0) 
plt.tight_layout()
plt.savefig('map analysis.png', dpi=300, bbox_inches='tight')






