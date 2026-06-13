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



########## interface data ############
interface = pyreadr.read_r("")[None]
cells_new = interface.copy()

cells_new['cell_type'].unique()


##############  Super-colocation ################
cells = cells_new.copy()
cells= cells.reset_index(drop=True)
cells_new["cell_id"] = (cells_new["cell_type"].astype(str) + "-" + (cells_new.groupby("cell_type").cumcount() + 1).astype(str))

patientIDs = cells['patientID'].unique().tolist()
imageIDs = cells['imageID'].unique().tolist()
patient_type = cells[['patientID', 'responder']].drop_duplicates().reset_index(drop=True)
image_type = cells[['imageID', 'responder']].drop_duplicates().reset_index(drop=True)


def process_fov(cells, imageID, features):
    fov = cells[cells['imageID']==imageID]
    cells_dict = {}
    for i in features:
        cell_subset = fov[fov['cell_type'] == i].copy()
        cells_dict[i] = cell_subset
    return cells_dict


features = cells['cell_type'].unique().tolist()
cells_dict = process_fov(cells, 'Mel30_001', features)
pair_list = list(permutations(features, 2))



def get_neighbors(cells_dict, distance=20, anchor='Tumor'):
    d = distance
    anchor_neighbors = cells_dict[anchor]
    anchor_coords = anchor_neighbors[['X', 'Y']].values
    features = list(cells_dict.keys())
    surround_features = [f for f in features if f != anchor]
    for i in surround_features: 
        cell_subset = cells_dict[i]
        feature_tree = cKDTree(cell_subset[['X', 'Y']].values)
        neighbor_indices_feature = feature_tree.query_ball_point(anchor_coords, r=d)
        anchor_neighbors[f'Num of {i} neighbors'] = [len(indices) for indices in neighbor_indices_feature]
        anchor_neighbors[f'Have {i} neighbors']=(anchor_neighbors[f'Num of {i} neighbors'] > 0).astype(int)
    return anchor_neighbors


anchor_neighbors = get_neighbors(cells_dict, distance=20, anchor='Tumor')


def get_mat_size(anchor_neighbors, features, anchor):
    mat_size = 0
    surround_features = [f for f in features if f != anchor]
    for i in surround_features:
        if anchor_neighbors[f'Num of {i} neighbors'].max() > mat_size:
            mat_size = anchor_neighbors[f'Num of {i} neighbors'].max()
    mat_size+=1
    return int(mat_size)

mat_size = get_mat_size(anchor_neighbors, features, anchor = 'Tumor')


def get_distributions_1d(
        anchor_neighbors,
        feature,     
        mat_size,
        responder = None,
        min_val = None     
    ):


    col = f'Num of {feature} neighbors'

    filtered = anchor_neighbors.loc[anchor_neighbors[col] >= min_val]

    mat_range = np.arange(min_val, mat_size)

    count = (
        filtered[col]
        .value_counts()
        .reindex(mat_range, fill_value=0)
        .sort_index())


    total = count.to_numpy().sum()
    n = len(mat_range)


    if total == 0:
        prob = pd.Series(np.zeros(n, dtype=float), index=mat_range)
        cumu_prob = pd.Series(np.zeros(n, dtype=float), index=mat_range)

        return count, prob, cumu_prob

    prob = count / total

    survival_counts = np.array([(filtered[col] >= k).sum() for k in mat_range])
    cumu_prob = pd.Series(survival_counts / total, index=mat_range)

    return count, prob, cumu_prob


count, prob, cumu_prob = get_distributions_1d(anchor_neighbors = anchor_neighbors, feature = 'B cell', mat_size = mat_size, responder='Responder', min_val=0)


prob_df = prob.to_frame()




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


def trim_distributions(
    resp_prob,
    nonresp_prob,
    trim_mode="trailing"   
):


    idx = resp_prob.index.union(nonresp_prob.index)
    resp_prob = resp_prob.reindex(idx, fill_value=0)
    nonresp_prob = nonresp_prob.reindex(idx, fill_value=0)

    if trim_mode == "none":
        return resp_prob, nonresp_prob

    both_zero = (resp_prob == 0) & (nonresp_prob == 0)

    if trim_mode == "all":
        resp_prob = resp_prob[~both_zero]
        nonresp_prob = nonresp_prob[~both_zero]
        return resp_prob, nonresp_prob

    if trim_mode == "trailing":
        if (~both_zero).any():
            last_valid = (~both_zero)[~both_zero].index[-1]
            resp_prob = resp_prob.loc[:last_valid]
            nonresp_prob = nonresp_prob.loc[:last_valid]
        return resp_prob, nonresp_prob

    raise ValueError("trim_mode must be one of {'none', 'trailing', 'all'}")




def _resolve_text_collisions(ax, texts, y_step=2, max_shifts=50):

    fig = ax.figure
    fig.canvas.draw()  
    renderer = fig.canvas.get_renderer()

    placed_bboxes = []
    for t in texts:
        if not t.get_visible():
            continue

        for _ in range(max_shifts):
            bbox = t.get_window_extent(renderer=renderer).expanded(1.02, 1.05)
            overlap = any(bbox.overlaps(b) for b in placed_bboxes)
            if not overlap:
                placed_bboxes.append(bbox)
                break

            x_off, y_off = t.get_position()
            t.set_position((x_off, y_off + y_step))

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()




def annotate_dots_collision(
    ax, x, y,
    fmt="{:.3f}",
    fontsize=9,
    base_offset=5,
    y_step=2,
    max_shifts=50,
    skip_zeros=True
):
    texts = []
    for xi, yi in zip(x, y):
        if skip_zeros and yi == 0:
            continue

        t = ax.annotate(
            fmt.format(yi),
            xy=(xi, yi),
            xytext=(0, base_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize
        )
        texts.append(t)

    _resolve_text_collisions(ax, texts, y_step=y_step, max_shifts=max_shifts)
    return texts




def plot_probability_distributions_lines(
    resp_prob,
    nonresp_prob,
    feature_name,
    anchor,
    figure_type,
    y_label,
    n_resp,
    n_nonresp,
    trim_mode="trailing",
    figsize=(6, 4),
    marker='o',
    markersize=4,
    linewidth=2,
    grid_alpha=0.3,
    annotate=True,
    value_fmt="{:.3f}",
    show=True
):
    resp_prob, nonresp_prob = trim_distributions(
        resp_prob, nonresp_prob, trim_mode
    )
    
    x = resp_prob.index

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    ax.plot(x, resp_prob.values, label=f"Responder (n={n_resp})",
            linewidth=linewidth, marker=marker, markersize=markersize)

    ax.plot(x, nonresp_prob.values, label=f"Non-Responder (n={n_nonresp})",
            linewidth=linewidth, marker=marker, markersize=markersize)

    if annotate:
        annotate_dots_collision(ax, x, resp_prob.values, fmt=value_fmt)
        annotate_dots_collision(ax, x, nonresp_prob.values, fmt=value_fmt)

    ax.set_xticks(x)
    ax.set_title(f"{figure_type} of {feature_name} Around {anchor}")
    ax.set_xlabel(f"Num of {feature_name} neighbors around {anchor}")
    ax.set_ylabel("Probability")

    ax.legend()
    ax.grid(True, alpha=grid_alpha)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax





def annotate_bars_collision(
    ax,
    fmt="{:.3f}",
    fontsize=9,
    base_offset=2,
    y_step=2,
    max_shifts=50,
    skip_zeros=True
):
    texts = []
    for bar in ax.patches:
        h = bar.get_height()
        if skip_zeros and h == 0:
            continue

        x = bar.get_x() + bar.get_width() / 2
        y = h

        t = ax.annotate(
            fmt.format(h),
            xy=(x, y),
            xytext=(0, base_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize
        )
        texts.append(t)

    _resolve_text_collisions(ax, texts, y_step=y_step, max_shifts=max_shifts)
    return texts


def annotate_bars(ax, fmt="{:.3f}", fontsize=9, padding=2):

    for bar in ax.patches:
        height = bar.get_height()
        if height == 0:
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize
        )


def plot_probability_distributions_bars(
    resp_prob,
    nonresp_prob,
    feature_name,
    anchor,
    figure_type,
    y_label,
    n_resp,
    n_nonresp,
    trim_mode="trailing",
    figsize=(6, 4),
    bar_width=0.4,
    alpha=0.9,
    grid_alpha=0.3,
    annotate=True,
    value_fmt="{:.3f}",
    show=True
):
    resp_prob, nonresp_prob = trim_distributions(
        resp_prob, nonresp_prob, trim_mode
    )
    

    
    x = resp_prob.index.to_numpy()
    x_pos = np.arange(len(x))

    fig, ax = plt.subplots(figsize=figsize, dpi=400)

    ax.bar(x_pos - bar_width / 2, resp_prob.values,
           width=bar_width, label=f"Responder (n={n_resp})", alpha=alpha)

    ax.bar(x_pos + bar_width / 2, nonresp_prob.values,
           width=bar_width, label=f"Non-Responder (n={n_nonresp})", alpha=alpha)

    if annotate:
        annotate_bars_collision(ax, fmt=value_fmt)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)

    ax.set_xlabel(f"Num of {feature_name} neighbors around {anchor}", fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    ax.legend()
    ax.grid(axis="y", alpha=grid_alpha)
    
    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.legend(fontsize=16)
    
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax





def get_js_distance(cells, features, neighbor_dist=20, min_val = 0, anchor='Tumor', plots=['count', 'prob', 'survival']):
    anchor_data_lst = []
    imageIDs = cells['imageID'].unique().tolist()
    for i in imageIDs:
        cells_dict = process_fov(cells, i, features)
        fov_anchor_neighbor = get_neighbors(cells_dict, distance = neighbor_dist, anchor=anchor)
        anchor_data_lst.append(fov_anchor_neighbor)
    anchor_neighbors = pd.concat(anchor_data_lst, ignore_index=True)
    
    mat_size = get_mat_size(anchor_neighbors, features, anchor)
    
    anchor_neigh_resp = anchor_neighbors[anchor_neighbors['responder'] == 1]
    anchor_neigh_nonresp = anchor_neighbors[anchor_neighbors['responder'] == 0]
    
    n_resp = len(anchor_neigh_resp)
    n_nonresp = len(anchor_neigh_nonresp)
    
    js_dist = {}
    surround_features = [f for f in features if f != anchor]
    for i in surround_features:
        resp_count, resp_prob, resp_cumu_prob = get_distributions_1d(anchor_neigh_resp, i, mat_size, 'Responder', min_val)
        nonresp_count, nonresp_prob, nonresp_cumu_prob = get_distributions_1d(anchor_neigh_nonresp, i, mat_size, 'Non-Responder', min_val)
        js = safe_js_distance(resp_prob, nonresp_prob, eps=1e-12)
        print(f'JS-Distance of {i} is {js}')
        js_dist[i] = js
        
        if "count" in plots:
            plot_probability_distributions_bars(resp_prob=resp_count, nonresp_prob=nonresp_count, feature_name=i, anchor = anchor, figure_type = "Raw Count", y_label = 'Count', n_resp = n_resp, n_nonresp = n_nonresp, annotate=False)
        
        if "prob" in plots:
            plot_probability_distributions_bars(resp_prob=resp_prob, nonresp_prob=nonresp_prob, feature_name=i, anchor = anchor, figure_type = "Probability Distribution", y_label = 'Probability', n_resp = n_resp, n_nonresp = n_nonresp, annotate=False)

        if "survival" in plots:
            plot_probability_distributions_bars(resp_prob=resp_cumu_prob, nonresp_prob=nonresp_cumu_prob, feature_name=i, anchor=anchor,figure_type = "Survival Function", y_label = "Probability", n_resp = n_resp, n_nonresp = n_nonresp, annotate=False)
        
        #plot_probability_distributions_lines(resp_prob=resp_prob, nonresp_prob=nonresp_prob, feature_name=i, anchor = anchor, annotate=False)
        #plot_probability_distributions_lines(resp_prob=resp_cumu_prob, nonresp_prob=nonresp_cumu_prob, feature_name=i, anchor=anchor, annotate=False)

    return js_dist




cells = cells_new.copy()

features = cells['cell_type'].unique().tolist()

js_results = get_js_distance(cells, features, neighbor_dist=20, min_val = 0, anchor='Tumor', plots = ['prob'])


supercolocation_js = pd.DataFrame(list(js_results.items()), columns=['cell_combos', 'js_distance'])
supercolocation_js = supercolocation_js.sort_values(by='js_distance', ascending=False).reset_index(drop=True)



#############  Visualization  ###########


def plot_js_distance(js_ranking):

    plt.figure(figsize=(8, 5), dpi=400)  

    plt.bar(js_ranking['cell_combos'], js_ranking['js_distance'], color='blue', width=0.6)

    plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=1, color='black')
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=14)
    plt.xlabel("Surround Cell Type", fontsize=18)
    plt.ylabel("JS Distance", fontsize=18)
    #plt.title("JS Distance by Surround Cell Type", fontsize=18)

    plt.tight_layout()
    plt.show()


plot_js_distance(supercolocation_js)




