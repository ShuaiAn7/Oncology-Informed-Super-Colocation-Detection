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
from matplotlib import gridspec




interface = pyreadr.read_r("")[None]
cells = interface.copy()


cell_types = cells['cell_type'].unique().tolist()
values_to_remove = ['Tumor', 'Tumor Cell']
cell_types = [x for x in cell_types if x not in values_to_remove]

patientIDs = cells['patientID'].unique().tolist()
imageIDs = cells['imageID'].unique().tolist()
patient_type = cells[['patientID', 'responder']].drop_duplicates().reset_index(drop=True)
responder_IDs = patient_type.loc[patient_type['responder'] == 1, 'patientID'].tolist()
nonresponder_IDs = patient_type.loc[patient_type['responder'] == 0, 'patientID'].tolist()


def process_fov(cells, imageID, features):
    fov = cells[cells['imageID']==imageID]

    cells_dict = {}
    cells_dict['Tumor'] = fov[fov['cell_type'] == 'Tumor'].copy()
    for i in features:
        cell_subset = fov[fov['cell_type'] == i].copy()
        cells_dict[i] = cell_subset
    return cells_dict



def get_neighbors(cells_dict, distance=10):
    d = distance
    fov_tumor = cells_dict['Tumor']
    tumor_coords = fov_tumor[['X', 'Y']].values
    feature_list = list(cells_dict.keys())
    feature_list.remove('Tumor')
    for i in feature_list: 
        cell_subset = cells_dict[i]
        feature_tree = cKDTree(cell_subset[['X', 'Y']].values)
        neighbor_indices_feature = feature_tree.query_ball_point(tumor_coords, r=d)
        fov_tumor[f'Num of {i} neighbors'] = [len(indices) for indices in neighbor_indices_feature]
        fov_tumor[f'Have {i} neighbors']=(fov_tumor[f'Num of {i} neighbors'] > 0).astype(int)
    return fov_tumor



def get_mat_size(tumor_neighbors, features):
    mat_size = 0
    for i in features:
        if tumor_neighbors[f'Num of {i} neighbors'].max() > mat_size:
            mat_size = tumor_neighbors[f'Num of {i} neighbors'].max()
    mat_size+=1
    return int(mat_size)



def plot_matrix(df, title="", xlabel="", ylabel="", cmap="viridis",
                            float_fmt=".3f"):
    data = df.values
    nrows, ncols = df.shape

    cell_size = 0.65
    min_size = 4
    max_size = 20
    fig_width  = np.clip(ncols * cell_size, min_size, max_size)
    fig_height = np.clip(nrows * cell_size, min_size, max_size)
    plt.figure(figsize=(fig_width, fig_height))

    is_integer_matrix = np.all(np.equal(np.mod(data, 1), 0))

    cmap_obj = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    plt.imshow(data, origin='upper', aspect='auto', cmap=cmap_obj, norm=norm)
    plt.colorbar(label='Value')

    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.xticks(np.arange(ncols), df.columns)
    plt.yticks(np.arange(nrows), df.index)

    for i in range(nrows):
        for j in range(ncols):
            value = data[i, j]
            brightness = norm(value)
            text_color = "white" if brightness < 0.45 else "black"

            if is_integer_matrix:
                text = format(int(value), "d")
            else:
                rounded = round(value, 3)
                if rounded == 0:
                    text = "0"
                else:
                    text = format(value, float_fmt)

            plt.text(
                j, i, text,
                ha="center", va="center",
                color=text_color, fontsize=9, fontweight='bold'
            )

    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, pad=35, fontsize=13)
    plt.tight_layout()
    plt.show()


def get_distributions(
        fov_tumor_neighbors,
        features,
        mat_size,
        responder,
        row_cond,
        col_cond):

    col_x = f'Num of {features[0]} neighbors'
    col_y = f'Num of {features[1]} neighbors'

    filtered = fov_tumor_neighbors.loc[
        (fov_tumor_neighbors[col_x] >= row_cond) &
        (fov_tumor_neighbors[col_y] >= col_cond)
    ]

    group_type_counts = (
        filtered
        .groupby([col_x, col_y])
        .size()
        .reset_index(name='count'))

    mat_range_x = np.arange(row_cond, mat_size)
    mat_range_y = np.arange(col_cond, mat_size)

    matrix_count_disjoint = (
        group_type_counts.pivot_table(
            index=col_x,
            columns=col_y,
            values='count',
            aggfunc='sum',
            fill_value=0
        )
        .reindex(index=mat_range_x, columns=mat_range_y, fill_value=0)
        .astype(int))


    total = matrix_count_disjoint.to_numpy().sum()
    nx, ny = len(mat_range_x), len(mat_range_y)

    if total == 0:
        matrix_prob = pd.DataFrame(
            np.zeros((nx, ny)),
            index=mat_range_x,
            columns=mat_range_y)

        return matrix_prob, matrix_count_disjoint

    matrix_prob = matrix_count_disjoint / total
    
    matrix = np.zeros((nx, ny), dtype=int)

    for ix, i in enumerate(mat_range_x):
        for jy, j in enumerate(mat_range_y):
            matrix[ix, jy] = (
                (filtered[col_x] >= i) &
                (filtered[col_y] >= j)
            ).sum()

    matrix_cumu_prob = pd.DataFrame(matrix, index=mat_range_x, columns=mat_range_y) / total
    
    
    return matrix_count_disjoint, matrix_prob, matrix_cumu_prob




def safe_js_distance(matA, matB, eps=1e-12):
    P = np.asarray(matA, float).ravel()
    Q = np.asarray(matB, float).ravel()

    sumP = P.sum()
    sumQ = Q.sum()

    if sumP == 0 and sumQ == 0:
        return 0.0

    if sumP == 0 or sumQ == 0:
        return 1.0

    P = P + eps
    Q = Q + eps
    P = P / P.sum()
    Q = Q / Q.sum()

    return jensenshannon(P, Q, base=2)





def plot_two_matrices(df_left,
                      df_right,
                      left_title="Responder",
                      right_title="Non-responder",
                      xlabel="Num of feature 2 neighbors",
                      ylabel="Num of feature 1 neighbors",
                      cmap="viridis",
                      float_fmt=".3f"):

    data_right = df_right.values
    data_left = df_left.values

    if data_left.shape != data_right.shape:
        raise ValueError("df_left and df_right must have the same shape")

    nrows, ncols = df_left.shape

    cell_size = 0.65
    min_size = 4
    max_size = 30
    fig_width  = np.clip(ncols * cell_size * 2.0, min_size, max_size)
    fig_height = np.clip(nrows * cell_size * 1.3, min_size, max_size)

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[1, 0.04],  
        width_ratios=[1, 1],
        hspace=0.1,
        wspace=0.1
    )

    ax_left  = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    cax      = fig.add_subplot(gs[1, :])   

    global_min = min(data_left.min(), data_right.min())
    global_max = max(data_left.max(), data_right.max())
    norm = mpl.colors.Normalize(vmin=global_min, vmax=global_max)
    cmap_obj = plt.get_cmap(cmap)

    def _plot_single(ax, df, title):
        data = df.values
        is_integer_matrix = np.all(np.equal(np.mod(data, 1), 0))

        im = ax.imshow(data, cmap=cmap_obj, norm=norm,
                       origin="upper", aspect="auto")

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel(xlabel, fontsize=13, labelpad=12)

        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_xticklabels(df.columns)
        ax.set_yticks(np.arange(df.shape[0]))
        ax.set_yticklabels(df.index)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                value = data[i, j]
                brightness = norm(value)
                text_color = "white" if brightness < 0.45 else "black"

                if is_integer_matrix:
                    text = str(int(value))
                else:
                    rounded = round(value, 3)
                    text = "0" if rounded == 0 else format(value, float_fmt)

                ax.text(j, i, text,
                        ha="center", va="center",
                        color=text_color, fontsize=9, fontweight='bold')

        ax.set_title(title, fontsize=13, pad=25)
        ax.set_ylabel(ylabel, fontsize=13)

        return im

    im_left  = _plot_single(ax_left, df_left,  left_title)
    #im_right = _plot_single(ax_right, df_right, right_title)
    
    ax_right.set_ylabel("")

    cbar = fig.colorbar(im_left, cax=cax, orientation="horizontal")
    cbar.set_label("Probability", fontsize=13)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    
    
    plt.show()





def get_js_distance(cells, features, neighbor_dist=10, row_cond=0, col_cond=0):
    tumor_data_lst = []
    for i in imageIDs:
        cells_dict = process_fov(cells, i, features)
        fov_tumor_neighbor = get_neighbors(cells_dict, distance = neighbor_dist)
        tumor_data_lst.append(fov_tumor_neighbor)
    tumor_neighbors = pd.concat(tumor_data_lst, ignore_index=True)
    
    mat_size = get_mat_size(tumor_neighbors, features)
    
    tumor_neigh_resp = tumor_neighbors[tumor_neighbors['responder'] == 1]
    resp_count, resp_prob, resp_cumu_prob = get_distributions(tumor_neigh_resp, features, mat_size, 'Responder', row_cond, col_cond)
    

    tumor_neigh_nonresp = tumor_neighbors[tumor_neighbors['responder'] == 0]
    nonresp_count, nonresp_prob, nonresp_cumu_prob = get_distributions(tumor_neigh_nonresp, features, mat_size, 'Non-Responder', row_cond, col_cond)
    
    feat1, feat2 = features
    xlabel = f"Num of {feat2} neighbors"
    ylabel = f"Num of {feat1} neighbors"


    plot_two_matrices(
        resp_count,
        nonresp_count,
        left_title="Responder Cell Count",
        right_title="Non-responder Cell Count",
        xlabel=xlabel,
        ylabel=ylabel,
        cmap="viridis",
        float_fmt=".3f"
    )
    


    plot_two_matrices(
        resp_prob,
        nonresp_prob,
        left_title="Responder probability",
        right_title="Non-responder probability",
        xlabel=xlabel,
        ylabel=ylabel,
        cmap="viridis",
        float_fmt=".3f"
    )
    
    plot_two_matrices(
        resp_cumu_prob,
        nonresp_cumu_prob,
        left_title="Responder Survival Function",
        right_title="Non-responder Survival Function",
        xlabel=xlabel,
        ylabel=ylabel,
        cmap="viridis",
        float_fmt=".3f"
    )
    

    
    js_dist = safe_js_distance(nonresp_prob, resp_prob)
    print(f"The JS-Distance between features {features} is {js_dist}")
    return js_dist



##############  All combos #################

combos = list(combinations(cell_types, 2))   


js_results = {}
for i in combos:
    js = get_js_distance(cells, i, neighbor_dist=20, row_cond=0, col_cond=0)
    js_results[i] = js

supercolocation_js = pd.DataFrame(list(js_results.items()), columns=['cell_combos', 'js_distance'])
supercolocation_js = supercolocation_js.sort_values(by='js_distance', ascending=False).reset_index(drop=True)
supercolocation_js['cell_combos'] = supercolocation_js['cell_combos'].apply(lambda t: ('Tumor',) + t)

supercolocation_js['Modulators'] = supercolocation_js['cell_combos'].apply(lambda tup: " ".join(x for x in tup if x not in ("CTL", "Tumor")))
cols = list(supercolocation_js.columns)
idx = cols.index('cell_combos')
cols.insert(idx + 1, cols.pop(cols.index('Modulators')))
supercolocation_js = supercolocation_js[cols]

supercolocation_js.to_csv('', index=False)


#############  Visualization  ###########


def plot_js_distance(js_ranking):

    plt.figure(figsize=(8, 5), dpi=300) 

    plt.bar(js_ranking['Modulators'], js_ranking['js_distance'], color='blue', width=0.6)

    plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=1, color='black')
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Modulators", fontsize=14)
    plt.ylabel("JS Distance", fontsize=14)
    plt.title("JS Distance by Modulators", fontsize=18)

    plt.tight_layout()
    plt.show()


plot_js_distance(supercolocation_js)

