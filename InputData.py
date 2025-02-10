import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random
from sklearn.impute import KNNImputer
import PySimpleGUI as sg
import seaborn as sns

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

plt.rcParams['figure.dpi'] = 300

###############################################################################
# KNN Imputation
###############################################################################
def knn_impute_data(df, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)

###############################################################################
# Data Preprocessing
###############################################################################
def preprocess_data(predictor_files, results_file=None, is_ranked=False, no_results=False):
    """
    Reads predictor CSVs and optionally a results CSV.
    Intersects on (drugs, individuals).
    Ranks them if is_ranked=False. Returns all relevant dataframes.
    """
    data_dict = {}
    predictor_dfs = []
    predictor_names = []

    # Read each predictor file
    for file_path in predictor_files:
        df = pd.read_csv(file_path, index_col=0)
        df.columns = df.columns.str.replace('.', '-', regex=False)
        if df.index[0] == "DRUG_NAME":
            df = df.drop(df.index[0])
        predictor_dfs.append(df)
        predictor_names.append(os.path.splitext(os.path.basename(file_path))[0])

    # Intersection of drugs and individuals among predictors
    common_drugs = set(predictor_dfs[0].index)
    common_individuals = set(predictor_dfs[0].columns)
    for df in predictor_dfs[1:]:
        common_drugs &= set(df.index)
        common_individuals &= set(df.columns)

    df_results_ranked = None
    if not no_results and results_file is not None:
        df_results = pd.read_csv(results_file, index_col=0)
        df_results.columns = df_results.columns.str.replace('.', '-', regex=False)
        if df_results.index[0] == "DRUG_NAME":
            df_results = df_results.drop(df_results.index[0])

        common_drugs &= set(df_results.index)
        common_individuals &= set(df_results.columns)

        common_drugs = sorted(list(common_drugs))
        common_individuals = sorted(list(common_individuals))
        df_results = df_results.loc[common_drugs, common_individuals]
    else:
        common_drugs = sorted(list(common_drugs))
        common_individuals = sorted(list(common_individuals))
        df_results = None

    # Align each predictor
    aligned_predictor_dfs = [df.loc[common_drugs, common_individuals] for df in predictor_dfs]

    if not is_ranked:
        ranked_predictor_dfs = []
        for df in aligned_predictor_dfs:
            df = df.apply(pd.to_numeric, errors='coerce')
            df_imputed = knn_impute_data(df)
            df_ranked = df_imputed.rank(axis=0, method='min', na_option='keep').astype(int)
            ranked_predictor_dfs.append(df_ranked)
    else:
        ranked_predictor_dfs = []
        for df in aligned_predictor_dfs:
            df_ranked = df.rank(axis=0, method='min', na_option='keep').astype(int)
            ranked_predictor_dfs.append(df_ranked)

    if df_results is not None:
        if not is_ranked:
            df_results = df_results.apply(pd.to_numeric, errors='coerce')
            df_results_imputed = knn_impute_data(df_results)
            df_results_ranked = df_results_imputed.rank(axis=0, method='min', na_option='keep').astype(int)
        else:
            df_results_ranked = df_results.rank(axis=0, method='min', na_option='keep').astype(int)

    for idx, df in enumerate(ranked_predictor_dfs):
        data_dict[f"Predictor{idx}"] = df.values.tolist()

    return data_dict, common_drugs, common_individuals, df_results_ranked, ranked_predictor_dfs, predictor_names

###############################################################################
# Simulation Helpers
###############################################################################
def RandomDataGenerator(num_individuals, num_drugs, num_predictors):
    """
    Generates random ranking data for simulation (GoldStandard + noisy Predictors).
    """
    temp_sim_data = {}
    counter = 0

    unique_ints = np.arange(1, num_drugs + 1)
    gold_standard = np.empty((num_drugs, num_individuals), dtype=int)
    for i in range(num_individuals):
        np.random.shuffle(unique_ints)
        gold_standard[:, i] = unique_ints
    temp_sim_data[f"Predictor{counter}"] = gold_standard.copy()

    for counter in range(1, num_predictors):
        degraded_predictor = gold_standard.copy()
        num_swaps = int(0.1 * num_drugs * num_individuals)  # 10% noise
        for _ in range(num_swaps):
            ind = random.randint(0, num_individuals - 1)
            drug1, drug2 = random.sample(range(num_drugs), 2)
            degraded_predictor[drug1, ind], degraded_predictor[drug2, ind] = (
                degraded_predictor[drug2, ind],
                degraded_predictor[drug1, ind],
            )
        temp_sim_data[f"Predictor{counter}"] = degraded_predictor

    temp_sim_data["GoldStandard"] = gold_standard
    return temp_sim_data, gold_standard

###############################################################################
# Stats Helpers
###############################################################################
def compute_drug_stds(predictor_dfs):
    """
    Computes the STD of drug ranks across all individuals and predictors.
    """
    rankings = [df.values for df in predictor_dfs]
    rankings = np.stack(rankings, axis=-1)  # (num_drugs, num_individuals, num_predictors)
    drug_stds = np.std(rankings, axis=(1, 2))
    return pd.Series(drug_stds, index=predictor_dfs[0].index)

def compute_individual_stds(predictor_dfs):
    """
    Computes the STD of individual ranks across all drugs and predictors.
    """
    rankings = [df.values for df in predictor_dfs]
    rankings = np.stack(rankings, axis=-1)
    individual_stds = np.std(rankings, axis=(0, 2))
    return pd.Series(individual_stds, index=predictor_dfs[0].columns)

def pearson_correlation(x, y):
    """
    Simple correlation function (no checks).
    """
    return np.corrcoef(x, y)[0, 1]

def calculate_scores_full(df_results_ranked, predictor_dfs):
    """
    Returns L2 distance measure for each predictor.
    """
    scores = []
    for df in predictor_dfs:
        diff = df_results_ranked - df
        score = np.nansum(diff ** 2)
        scores.append(score)
    return np.array(scores)

###############################################################################
# Simulation: find correlation subsets
###############################################################################
def run_project_simulation(num_individuals, num_drugs, num_predictors,
                          min_correlation, N,
                          common_drugs, common_individuals,
                          predictor_names, num_iterations):
    all_subsets = []
    correlations_matrix = None

    for iteration in range(num_iterations):
        temp_sim_data, gold_standard = RandomDataGenerator(num_individuals, num_drugs, num_predictors)

        predictor_dfs_sim = []
        predictor_names_sim = []
        for key in temp_sim_data.keys():
            if key != 'GoldStandard':
                df = pd.DataFrame(temp_sim_data[key],
                                  index=common_drugs[:num_drugs],
                                  columns=common_individuals[:num_individuals])
                predictor_dfs_sim.append(df)
                predictor_names_sim.append(key)

        gold_standard_df = pd.DataFrame(
            temp_sim_data['GoldStandard'],
            index=common_drugs[:num_drugs],
            columns=common_individuals[:num_individuals]
        )

        drug_stds = compute_drug_stds(predictor_dfs_sim)
        sorted_drugs = drug_stds.sort_values(ascending=False).index.tolist()

        individual_stds = compute_individual_stds(predictor_dfs_sim)
        sorted_individuals = individual_stds.sort_values(ascending=False).index.tolist()

        max_num_drugs = num_drugs
        max_num_individuals = num_individuals
        correlations_matrix = np.zeros((max_num_drugs, max_num_individuals))

        for num_drug in range(1, max_num_drugs + 1):
            top_drugs = sorted_drugs[:num_drug]
            for num_indiv in range(1, max_num_individuals + 1):
                top_individuals = sorted_individuals[:num_indiv]

                gold_subset = gold_standard_df.loc[top_drugs, top_individuals]
                predictor_dfs_subset = [df.loc[top_drugs, top_individuals] for df in predictor_dfs_sim]

                gold_mean_rank = gold_subset.mean(axis=1).rank(method='min')
                corrs = []
                for predictor_df in predictor_dfs_subset:
                    p_mean_rank = predictor_df.mean(axis=1).rank(method='min')
                    corr_val = pearson_correlation(p_mean_rank.values, gold_mean_rank.values)
                    if np.isnan(corr_val):
                        corr_val = 0
                    corrs.append(corr_val)

                avg_correlation = np.mean(corrs)
                correlations_matrix[num_drug - 1, num_indiv - 1] = avg_correlation

                subset_info = {
                    'Drugs': top_drugs,
                    'Individuals': top_individuals,
                    'Correlation': avg_correlation,
                    'NumDrugs': num_drug,
                    'NumIndividuals': num_indiv
                }
                all_subsets.append(subset_info)

        # Only do 1 iteration typically
        break

    subsets_df = pd.DataFrame(all_subsets)
    grouped_subsets = subsets_df.groupby(['NumDrugs', 'NumIndividuals']).agg({
        'Correlation': 'mean',
        'Drugs': 'first',
        'Individuals': 'first'
    }).reset_index()

    valid_subsets = grouped_subsets[grouped_subsets['Correlation'] >= min_correlation]
    if valid_subsets.empty:
        num_drugs_first = None
        num_individuals_first = None
    else:
        valid_subsets = valid_subsets.sort_values(by=['NumDrugs', 'NumIndividuals'])
        top_subsets = valid_subsets.head(N)
        num_drugs_first = top_subsets.iloc[0]['NumDrugs']
        num_individuals_first = top_subsets.iloc[0]['NumIndividuals']

    return correlations_matrix, num_drugs_first, num_individuals_first, grouped_subsets

###############################################################################
# Optimize Ranking Correlation
###############################################################################
def optimize_ranking_correlation(df_results_ranked, predictor_dfs, min_correlation, N):
    full_data_scores = calculate_scores_full(df_results_ranked, predictor_dfs)
    full_data_ranking = np.argsort(full_data_scores) + 1  # rank 1-based

    num_drugs_total = len(df_results_ranked.index)
    num_individuals_total = len(df_results_ranked.columns)

    drug_stds = compute_drug_stds(predictor_dfs)
    sorted_drugs = drug_stds.sort_values(ascending=False).index.tolist()
    sorted_individuals = df_results_ranked.columns.tolist()

    correlations = np.zeros((num_drugs_total, num_individuals_total))
    drugs_used_dict, individuals_used_dict = {}, {}
    all_subsets = []

    for nd in range(1, num_drugs_total + 1):
        top_drugs = sorted_drugs[:nd]
        drugs_used_dict[nd] = top_drugs

        for ni in range(1, num_individuals_total + 1):
            top_individuals = sorted_individuals[:ni]
            individuals_used_dict[ni] = top_individuals

            df_results_subset = df_results_ranked.loc[top_drugs, top_individuals]
            predictor_dfs_subset = [df.loc[top_drugs, top_individuals] for df in predictor_dfs]

            subset_scores = calculate_scores_full(df_results_subset, predictor_dfs_subset)
            subset_ranking = np.argsort(subset_scores) + 1

            corr_val = pearson_correlation(subset_ranking, full_data_ranking)
            if np.isnan(corr_val):
                corr_val = 0

            correlations[nd - 1, ni - 1] = corr_val
            subset_info = {
                'NumDrugs': nd,
                'NumIndividuals': ni,
                'Correlation': corr_val,
                'Drugs': top_drugs,
                'Individuals': top_individuals
            }
            all_subsets.append(subset_info)

    subsets_df = pd.DataFrame(all_subsets)
    valid_subsets = subsets_df[subsets_df['Correlation'] >= min_correlation]

    if valid_subsets.empty:
        nd_first = None
        ni_first = None
    else:
        valid_subsets = valid_subsets.sort_values(by=['NumDrugs', 'NumIndividuals'])
        top_subsets = valid_subsets.head(N)
        nd_first = top_subsets.iloc[0]['NumDrugs']
        ni_first = top_subsets.iloc[0]['NumIndividuals']

    return correlations, nd_first, ni_first, subsets_df, drugs_used_dict, individuals_used_dict

###############################################################################
# Static Heatmaps
###############################################################################
def plot_static_heatmap(correlations_matrix, min_corr,
                        num_drugs_first, num_individuals_first,
                        subsets_df, N):
    """
    Plots a heatmap of correlations_matrix and a 10x10 zoom around the first subset.
    If the matrix is larger than 100 in either dimension, we uniformly sample
    down to 100 and remove annotations so the plot is clearer.
    """
    if correlations_matrix is None or correlations_matrix.size == 0:
        return

    max_plot_dim = 100
    n_drugs, n_inds = correlations_matrix.shape

    # Downsample if needed
    if n_drugs > max_plot_dim or n_inds > max_plot_dim:
        drg_indices = np.linspace(0, n_drugs - 1, max_plot_dim, dtype=int)
        ind_indices = np.linspace(0, n_inds - 1, max_plot_dim, dtype=int)
        matrix_to_plot = correlations_matrix[drg_indices][:, ind_indices]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matrix_to_plot,
                    cmap='Blues',
                    cbar=True,
                    square=True,
                    annot=False)  # No annotations to avoid clutter
        ax.set_title("Full Correlation Matrix (Downsampled)")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()
    else:
        # If smaller, we can annotate
        annot_data = np.empty_like(correlations_matrix, dtype=object)
        for r in range(correlations_matrix.shape[0]):
            for c in range(correlations_matrix.shape[1]):
                annot_data[r, c] = f"{correlations_matrix[r, c]:.2f}"

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlations_matrix,
                    cmap='Blues',
                    cbar=True,
                    square=True,
                    annot=annot_data,
                    fmt="",
                    xticklabels=False,
                    yticklabels=False,
                    ax=ax)
        ax.set_title("Full Correlation Matrix")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    # Zoom 10x10 around the first valid subset
    if num_drugs_first is not None and num_individuals_first is not None:
        idx_drug = num_drugs_first - 1
        idx_indiv = num_individuals_first - 1
        window_size = 10

        start_drug_idx = max(idx_drug - window_size // 2, 0)
        end_drug_idx = min(start_drug_idx + window_size, correlations_matrix.shape[0])
        start_indiv_idx = max(idx_indiv - window_size // 2, 0)
        end_indiv_idx = min(start_indiv_idx + window_size, correlations_matrix.shape[1])

        zoomed_data = correlations_matrix[start_drug_idx:end_drug_idx,
                                          start_indiv_idx:end_indiv_idx]

        if zoomed_data.size == 0:
            return

        annot_zoom = np.empty_like(zoomed_data, dtype=object)
        for r in range(zoomed_data.shape[0]):
            for c in range(zoomed_data.shape[1]):
                annot_zoom[r, c] = f"{zoomed_data[r, c]:.2f}"

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(zoomed_data,
                    cmap='Blues',
                    cbar=True,
                    square=True,
                    annot=annot_zoom,
                    fmt="",
                    xticklabels=range(start_indiv_idx+1, end_indiv_idx+1),
                    yticklabels=range(start_drug_idx+1, end_drug_idx+1),
                    ax=ax)
        ax.set_xlabel('Number of Individuals')
        ax.set_ylabel('Number of Drugs')
        ax.set_title(f"Zoom 10x10 Around Subset: {num_drugs_first} drugs, {num_individuals_first} individuals")
        ax.invert_yaxis()

        for r in range(zoomed_data.shape[0]):
            for c in range(zoomed_data.shape[1]):
                if zoomed_data[r, c] >= min_corr:
                    rect = plt.Rectangle((c, r), 1, 1,
                                         fill=False,
                                         edgecolor="white",
                                         linewidth=2)
                    ax.add_patch(rect)

        plt.tight_layout()
        plt.show()

###############################################################################
# Bar Chart & PCA
###############################################################################
def plot_bar_chart_with_significance(scores, predictor_labels, p_values):
    """
    Shows scaled L2-distance in a bar chart.
    """
    buffer = 0.1
    max_score = max(scores)
    min_score = min(scores)
    range_score = max_score - min_score
    if range_score != 0:
        scaled_scores = [
            1 - (buffer + (score - min_score) / range_score * (1 - 2 * buffer))
            for score in scores
        ]
    else:
        scaled_scores = [0.5] * len(scores)

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(scaled_scores))
    plt.bar(predictor_labels, scaled_scores, color=colors)
    plt.ylim([0, 1])
    plt.xlabel('Predictors')
    plt.ylabel('Score (Scaled)')
    plt.title('Predictor Scores')
    plt.tight_layout()
    plt.show()

def plot_histogram_for_individual_correlations(df_results_ranked, predictor_dfs, predictor_names):
    """
    For each predictor, plot bar chart of correlation with results per individual.
    """
    for idx, df_predictor in enumerate(predictor_dfs):
        individual_correlations = []
        individuals = df_results_ranked.columns
        for individual in individuals:
            corr = pearson_correlation(df_results_ranked[individual], df_predictor[individual])
            individual_correlations.append(corr)

        overall_corr = pearson_correlation(df_results_ranked.values.flatten(),
                                           df_predictor.values.flatten())
        median_corr = np.median(individual_correlations)
        avg_corr = np.mean(individual_correlations)

        plt.figure(figsize=(10, 4))
        plt.bar(individuals, individual_correlations, color='skyblue')
        plt.axhline(y=overall_corr, color='red', linestyle='--', label=f'Overall: {overall_corr:.2f}')
        plt.axhline(y=median_corr, color='green', linestyle='--', label=f'Median: {median_corr:.2f}')
        plt.axhline(y=avg_corr, color='orange', linestyle='--', label=f'Avg: {avg_corr:.2f}')
        plt.ylim([-1, 1])
        plt.title(f'{predictor_names[idx]} - Correlations')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()

def apply_pca_and_plot(df_results_ranked, predictor_dfs, predictor_names, has_results):
    """
    PCA where each column is one sample in #drugs-dimensional space.
    """
    all_data = []
    group_labels = []

    if has_results and df_results_ranked is not None:
        for col_name in df_results_ranked.columns:
            col_vector = df_results_ranked[col_name].values
            all_data.append(col_vector)
            group_labels.append("Results")

    for idx, df_pred in enumerate(predictor_dfs):
        for col_name in df_pred.columns:
            col_vector = df_pred[col_name].values
            all_data.append(col_vector)
            group_labels.append(predictor_names[idx])

    all_data = np.array(all_data)
    if all_data.size == 0:
        return

    mean_vals = all_data.mean(axis=0)
    std_vals = all_data.std(axis=0)
    std_vals[std_vals == 0] = 1e-9
    all_data_std = (all_data - mean_vals) / std_vals

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_data_std)
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100

    unique_groups = sorted(list(set(group_labels)))
    palette = sns.color_palette("Set2", len(unique_groups))
    color_map = {grp: palette[i] for i, grp in enumerate(unique_groups)}

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(group_labels):
        plt.scatter(pca_result[i, 0],
                    pca_result[i, 1],
                    color=color_map[label],
                    s=60,
                    alpha=0.75,
                    edgecolors='k')

    handles = []
    for grp in unique_groups:
        handles.append(plt.Line2D([], [], marker="o", color=color_map[grp],
                                  linestyle="", label=grp, markersize=10, markeredgecolor='k'))
    plt.legend(handles=handles, title="Data Source", loc='best')
    plt.xlabel(f"PC1 ({pc1_var:.2f}% Var)")
    plt.ylabel(f"PC2 ({pc2_var:.2f}% Var)")
    plt.title("PCA of Predictions vs. Results")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_predictor_relationship_with_results_heatmap(predictor_dfs, results_df, predictor_names):
    """
    Plots correlation among all predictors + results (if present).
    """
    all_dfs = []
    all_names = []

    if results_df is not None:
        all_dfs.append(results_df)
        all_names.append("Results")

    for i, df_pred in enumerate(predictor_dfs):
        all_dfs.append(df_pred)
        all_names.append(predictor_names[i])

    if len(all_dfs) == 0:
        return

    num_predictors = len(all_dfs)
    correlation_matrix = np.zeros((num_predictors, num_predictors))

    for i in range(num_predictors):
        for j in range(num_predictors):
            cval = pearson_correlation(all_dfs[i].values.flatten(),
                                       all_dfs[j].values.flatten())
            correlation_matrix[i, j] = cval

    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=all_names,
                yticklabels=all_names,
                cbar_kws={'label': 'Pearson Corr'})
    plt.title("Predictors-Results Correlation")
    plt.tight_layout()
    plt.show()

###############################################################################
# Permutation Test
###############################################################################
def permutation_test(df_results_ranked, predictor_dfs, n_permutations=10):
    actual_scores = calculate_scores_full(df_results_ranked, predictor_dfs)
    permuted_scores_distribution = np.zeros((n_permutations, len(predictor_dfs)))

    for perm_idx in range(n_permutations):
        permuted_results = df_results_ranked.apply(np.random.permutation, axis=0)
        permuted_scores = calculate_scores_full(permuted_results, predictor_dfs)
        permuted_scores_distribution[perm_idx] = permuted_scores

    p_values = []
    for i in range(len(actual_scores) - 1):
        for j in range(i + 1, len(actual_scores)):
            actual_diff = abs(actual_scores[i] - actual_scores[j])
            perm_diff_distribution = np.abs(permuted_scores_distribution[:, i]
                                            - permuted_scores_distribution[:, j])
            p_value = np.mean(perm_diff_distribution >= actual_diff)
            p_values.append((i, j, p_value))
    return p_values, actual_scores

###############################################################################
# Cost-Effectiveness
###############################################################################
def build_cost_dict(subsets_df):
    """
    Convert subsets (NumIndividuals, NumDrugs, Correlation) -> dict for dash.
    """
    data_dict = {}
    if subsets_df is None or subsets_df.empty:
        return data_dict
    for _, row in subsets_df.iterrows():
        key = f"i{int(row['NumIndividuals'])}d{int(row['NumDrugs'])}"
        data_dict[key] = float(row['Correlation'])
    return data_dict

def plot_cost_effectiveness_heatmap(data_dict, min_corr):
    """
    Builds a cost matrix from data_dict and plots it.
    If all are zero, no plot. Also does a 10x10 zoom around the max cell,
    but with annotations for the zoomed region.
    """
    if not data_dict:
        return

    subset_labels = list(data_dict.keys())
    x_values = sorted({int(lbl.split('i')[1].split('d')[0]) for lbl in subset_labels})
    y_values = sorted({int(lbl.split('d')[1]) for lbl in subset_labels})

    cost_matrix = np.zeros((len(y_values), len(x_values)))
    for lbl, corr_val in data_dict.items():
        i_val = int(lbl.split('i')[1].split('d')[0])
        d_val = int(lbl.split('d')[1])
        x_idx = x_values.index(i_val)
        y_idx = y_values.index(d_val)
        if corr_val >= min_corr:
            cost = (corr_val / (i_val + d_val)) * 100
            cost_matrix[y_idx, x_idx] = cost
        else:
            cost_matrix[y_idx, x_idx] = 0.0

    if np.allclose(cost_matrix, 0):
        return

    fig_width = min(20, len(x_values) / 2 + 2)
    fig_height = min(20, len(y_values) / 2 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    vmax = cost_matrix.max() * 1.05 if cost_matrix.max() > 0 else 1

    # Main cost heatmap (no annotation to avoid clutter)
    sns.heatmap(cost_matrix,
                cmap="YlOrRd",
                vmin=0,
                vmax=vmax,
                annot=False,
                xticklabels=x_values,
                yticklabels=y_values,
                ax=ax)
    ax.set_xlabel("Number of Individuals")
    ax.set_ylabel("Number of Drugs")
    ax.set_title("Cost-Effectiveness Heatmap")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    max_coord = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
    max_y, max_x = max_coord[0], max_coord[1]

    window_size = 10
    start_x_idx = max(max_x - window_size // 2, 0)
    end_x_idx = min(start_x_idx + window_size, cost_matrix.shape[1])
    start_y_idx = max(max_y - window_size // 2, 0)
    end_y_idx = min(start_y_idx + window_size, cost_matrix.shape[0])

    zoomed_cost = cost_matrix[start_y_idx:end_y_idx, start_x_idx:end_x_idx]
    if zoomed_cost.size == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax_zoom = zoomed_cost.max() * 1.05 if zoomed_cost.max() > 0 else 1

    x_labels = [x_values[x] for x in range(start_x_idx, end_x_idx)]
    y_labels = [y_values[y] for y in range(start_y_idx, end_y_idx)]

    # Zoomed heatmap with annotations
    sns.heatmap(zoomed_cost,
                cmap="YlOrRd",
                vmin=0,
                vmax=vmax_zoom,
                annot=True,
                fmt=".2f",
                xticklabels=x_labels,
                yticklabels=y_labels,
                ax=ax)
    ax.set_xlabel("Number of Individuals")
    ax.set_ylabel("Number of Drugs")
    ax.set_title("Zoomed 10x10 Around Most Cost-Effective")
    ax.invert_yaxis()

    flat_coords = [(r, c) for r in range(zoomed_cost.shape[0])
                   for c in range(zoomed_cost.shape[1])]
    sorted_by_val = sorted(flat_coords, key=lambda rc: zoomed_cost[rc[0], rc[1]], reverse=True)
    top10 = sorted_by_val[:10]
    for (r, c) in top10:
        rect = plt.Rectangle((c, r), 1, 1,
                             fill=False,
                             edgecolor="white",
                             linewidth=2)
        ax.add_patch(rect)

    plt.tight_layout()
    plt.show()

###############################################################################
# PySimpleGUI for file input
###############################################################################
def file_input_gui():
    layout = [
        [sg.Text('Select CSV files for each predictor (click "+" to add more):')],
        [sg.Column([
            [sg.Input(key='Predictor0'),
             sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))]
        ], key='PredictorsColumn')],
        [sg.Button('+', key='AddPredictor')],
        [sg.Checkbox('No results file (use simulation)', key='NoResults', default=False, enable_events=True)],
        [sg.Text('Select CSV file for results:', key='ResultsLabel', visible=True)],
        [sg.Input(key='Results', visible=True),
         sg.FileBrowse(file_types=(("CSV Files", "*.csv"),), key='ResultsBrowse', visible=True)],
        [sg.Checkbox('Data is already ranked', key='Ranked', default=False)],
        [sg.Text('Simulation Parameters:')],
        [sg.Text('Minimum correlation:', key='MinCorrLabel'),
         sg.InputText('0.7', key='MinCorrelation')],
        [sg.Text('Number of subsets to find (N):', key='NumSubsetsLabel'),
         sg.InputText('1', key='NumSubsets')],
        [sg.Text('Number of iterations:', key='NumIterationsLabel', visible=False),
         sg.InputText('1', key='NumIterations', visible=False)],
        [sg.Button('Submit'), sg.Button('Cancel')]
    ]

    window = sg.Window('Data Input', layout, finalize=True)
    predictor_keys = ['Predictor0']

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, 'Cancel'):
            window.close()
            return None, None, None, None, None, None, None

        if event == 'AddPredictor':
            new_key = f'Predictor{len(predictor_keys)}'
            predictor_keys.append(new_key)
            window.extend_layout(
                window['PredictorsColumn'],
                [[sg.Input(key=new_key),
                  sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))]]
            )
            window.refresh()

        if event == 'NoResults':
            if values['NoResults']:
                window['ResultsLabel'].update(visible=False)
                window['Results'].update(visible=False)
                window['ResultsBrowse'].update(visible=False)
                window['NumIterationsLabel'].update(visible=True)
                window['NumIterations'].update(visible=True)
            else:
                window['ResultsLabel'].update(visible=True)
                window['Results'].update(visible=True)
                window['ResultsBrowse'].update(visible=True)
                window['NumIterationsLabel'].update(visible=False)
                window['NumIterations'].update(visible=False)
            window.refresh()

        if event == 'Submit':
            predictor_files = [values[key] for key in predictor_keys if values[key]]
            is_ranked = values['Ranked']
            no_results = values['NoResults']

            if not predictor_files:
                sg.popup("Please select at least one predictor file.", title="Error")
                continue

            try:
                min_correlation = float(values['MinCorrelation'])
                num_subsets = int(values['NumSubsets'])
            except ValueError:
                sg.popup("Enter valid numbers for min correlation and subsets.", title="Error")
                continue

            if no_results:
                try:
                    num_iterations = int(values['NumIterations'])
                except ValueError:
                    sg.popup("Enter a valid integer for number of iterations.", title="Error")
                    continue
                results_file = None
            else:
                results_file = values['Results']
                if not results_file:
                    sg.popup("Select a results file or choose simulation option.", title="Error")
                    continue
                num_iterations = None

            window.close()
            return (predictor_files, results_file, is_ranked,
                    no_results, min_correlation, num_subsets, num_iterations)

###############################################################################
# Global placeholders for the Dash app
###############################################################################
drugs_used_dict_global = {}
individuals_used_dict_global = {}
data_dict_global = {}
min_corr_global = 0.7

###############################################################################
# Main
###############################################################################
def main():
    global drugs_used_dict_global
    global individuals_used_dict_global
    global data_dict_global
    global min_corr_global

    (predictor_files, results_file, is_ranked,
     no_results, min_correlation, num_subsets, num_iterations) = file_input_gui()

    if not predictor_files:
        return

    drugs_used_dict_global = {}
    individuals_used_dict_global = {}
    data_dict_global = {}
    min_corr_global = min_correlation

    try:
        (data_dict,
         common_drugs,
         common_individuals,
         df_results_ranked,
         ranked_predictor_dfs,
         predictor_names) = preprocess_data(
             predictor_files,
             results_file,
             is_ranked,
             no_results
        )

        if no_results:
            num_predictors = len(predictor_names)
            num_drugs = len(common_drugs)
            num_individuals = len(common_individuals)

            (correlations_matrix,
             num_drugs_first,
             num_individuals_first,
             subsets_df) = run_project_simulation(
                 num_individuals,
                 num_drugs,
                 num_predictors,
                 min_correlation,
                 num_subsets,
                 common_drugs,
                 common_individuals,
                 predictor_names,
                 num_iterations
            )

            if correlations_matrix is not None and correlations_matrix.size > 0:
                plot_static_heatmap(correlations_matrix,
                                    min_correlation,
                                    num_drugs_first,
                                    num_individuals_first,
                                    subsets_df,
                                    num_subsets)

            data_dict_global = build_cost_dict(subsets_df)
            plot_cost_effectiveness_heatmap(data_dict_global, min_correlation)

        else:
            has_results = (df_results_ranked is not None)

            p_values, full_data_scores = permutation_test(df_results_ranked, ranked_predictor_dfs)
            plot_bar_chart_with_significance(full_data_scores, predictor_names, p_values)
            plot_histogram_for_individual_correlations(df_results_ranked, ranked_predictor_dfs, predictor_names)
            apply_pca_and_plot(df_results_ranked, ranked_predictor_dfs, predictor_names, has_results)
            plot_predictor_relationship_with_results_heatmap(ranked_predictor_dfs, df_results_ranked, predictor_names)

            (correlations,
             nd_first,
             ni_first,
             subsets_df,
             drugs_used_dict,
             individuals_used_dict) = optimize_ranking_correlation(
                 df_results_ranked,
                 ranked_predictor_dfs,
                 min_correlation,
                 num_subsets
            )

            drugs_used_dict_global = drugs_used_dict
            individuals_used_dict_global = individuals_used_dict

            if correlations is not None and correlations.size > 0:
                plot_static_heatmap(correlations,
                                    min_correlation,
                                    nd_first,
                                    ni_first,
                                    subsets_df,
                                    num_subsets)

            data_dict_global = build_cost_dict(subsets_df)
            plot_cost_effectiveness_heatmap(data_dict_global, min_correlation)

    except Exception as e:
        print(f"Error occurred: {e}")

###############################################################################
# Dash App for Interactive Heatmaps
###############################################################################
def run_dash_app():
    """
    After calling main(), we have data_dict_global, min_corr_global,
    and dictionaries from the last optimization call.
    We'll plot a 10x10 window around the best correlation cell in correlation space,
    plus a cost-effectiveness version side by side.
    """
    subset_labels = list(data_dict_global.keys())
    if not subset_labels:
        print("No data available for the Dash app.")
        return

    max_i = 0
    max_d = 0
    for lbl in subset_labels:
        i = int(lbl.split('i')[1].split('d')[0])
        d = int(lbl.split('d')[1])
        if i > max_i:
            max_i = i
        if d > max_d:
            max_d = d

    # Build the correlation matrix
    corr_matrix_full = np.zeros((max_d, max_i))
    for lbl, val in data_dict_global.items():
        i = int(lbl.split('i')[1].split('d')[0])
        d = int(lbl.split('d')[1])
        corr_matrix_full[d - 1, i - 1] = val

    # Find best subset by correlation
    best_key = max(data_dict_global, key=data_dict_global.get)
    best_i = int(best_key.split('i')[1].split('d')[0])
    best_d = int(best_key.split('d')[1])

    window_size = 10

    start_i = best_i - window_size // 2
    end_i = start_i + window_size
    if start_i < 1:
        start_i = 1
        end_i = start_i + window_size
    if end_i > max_i:
        end_i = max_i
        start_i = max(end_i - window_size + 1, 1)

    start_d = best_d - window_size // 2
    end_d = start_d + window_size
    if start_d < 1:
        start_d = 1
        end_d = start_d + window_size
    if end_d > max_d:
        end_d = max_d
        start_d = max(end_d - window_size + 1, 1)

    # Submatrix for correlation
    corr_sub = corr_matrix_full[start_d-1:end_d-1, start_i-1:end_i-1]

    # Build cost submatrix using correlation sub for reference
    cost_sub = np.zeros_like(corr_sub)
    for r in range(corr_sub.shape[0]):
        for c in range(corr_sub.shape[1]):
            d_idx = start_d + r
            i_idx = start_i + c
            val_corr = corr_matrix_full[d_idx - 1, i_idx - 1]
            if val_corr >= min_corr_global:
                cost_val = (val_corr / (i_idx + d_idx)) * 100.0
            else:
                cost_val = 0.0
            cost_sub[r, c] = cost_val

    # Build x/y labels
    x_labels = [str(i) for i in range(start_i, end_i)]
    y_labels = [str(d) for d in range(start_d, end_d)]

    sub_width = corr_sub.shape[1]
    sub_height = corr_sub.shape[0]

    # Correlation figure
    # We'll create custom annotations from corr_sub
    annotations_corr = []
    zmin_corr = corr_sub.min()
    zmax_corr = corr_sub.max()
    for row in range(sub_height):
        for col in range(sub_width):
            val = corr_sub[row, col]
            text_color = "white" if val >= min_corr_global else "black"
            annotations_corr.append(dict(
                text=f"{val:.2f}",
                x=col,  # sub index
                y=row,  # sub index
                xref='x1',
                yref='y1',
                showarrow=False,
                font=dict(color=text_color, size=10)
            ))

    heatmap_corr = go.Heatmap(
        z=corr_sub,
        x=list(range(sub_width)),
        y=list(range(sub_height)),
        colorscale='Blues',
        showscale=True,
        zmin=zmin_corr,
        zmax=zmax_corr,
        colorbar=dict(title="Correlation")
    )

    layout_corr = go.Layout(
        title="Correlation Heatmap (10x10 Window)",
        annotations=annotations_corr,
        xaxis=dict(
            title='Number of Individuals',
            tickmode='array',
            tickvals=list(range(sub_width)),
            ticktext=x_labels,
            tickangle=-45
        ),
        yaxis=dict(
            title='Number of Drugs',
            tickmode='array',
            tickvals=list(range(sub_height)),
            ticktext=y_labels
        ),
    )
    fig_corr = go.Figure(data=[heatmap_corr], layout=layout_corr)

    # Cost figure
    annotations_cost = []
    zmin_cost = cost_sub.min()
    zmax_cost = cost_sub.max()
    for row in range(sub_height):
        for col in range(sub_width):
            val = cost_sub[row, col]
            corr_val = corr_sub[row, col]
            text_color = "white" if corr_val >= min_corr_global else "black"
            annotations_cost.append(dict(
                text=f"{val:.2f}",
                x=col,
                y=row,
                xref='x1',
                yref='y1',
                showarrow=False,
                font=dict(color=text_color, size=10)
            ))

    heatmap_cost = go.Heatmap(
        z=cost_sub,
        x=list(range(sub_width)),
        y=list(range(sub_height)),
        colorscale='YlOrRd',
        showscale=True,
        zmin=zmin_cost,
        zmax=zmax_cost if zmax_cost > 0 else 1,
        colorbar=dict(title="Cost-Effectiveness")
    )

    layout_cost = go.Layout(
        title="Cost-Effectiveness Heatmap (10x10 Window)",
        annotations=annotations_cost,
        xaxis=dict(
            title='Number of Individuals',
            tickmode='array',
            tickvals=list(range(sub_width)),
            ticktext=x_labels,
            tickangle=-45
        ),
        yaxis=dict(
            title='Number of Drugs',
            tickmode='array',
            tickvals=list(range(sub_height)),
            ticktext=y_labels
        ),
    )
    fig_cost = go.Figure(data=[heatmap_cost], layout=layout_cost)

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='heatmap_corr', figure=fig_corr, style={'display': 'inline-block', 'width': '45%'}),
            dcc.Graph(id='heatmap_cost', figure=fig_cost, style={'display': 'inline-block', 'width': '45%'})
        ]),
        html.Div(id='click-data-corr',
                 style={'whiteSpace': 'pre-line', 'padding': '20px', 'font-size': '16px',
                        'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div(id='click-data-cost',
                 style={'whiteSpace': 'pre-line', 'padding': '20px', 'font-size': '16px',
                        'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ])

    @app.callback(
        Output('click-data-corr', 'children'),
        [Input('heatmap_corr', 'clickData')]
    )
    def display_click_data_corr(clickData):
        if clickData is None:
            return "Click a cell in the Correlation Heatmap."
        point = clickData['points'][0]
        x_sub = int(point['x'])  # 0-based within the submatrix
        y_sub = int(point['y'])

        i_val = start_i + x_sub
        d_val = start_d + y_sub

        corr_val = corr_matrix_full[d_val - 1, i_val - 1]

        drugs = drugs_used_dict_global.get(d_val, [])
        individuals = individuals_used_dict_global.get(i_val, [])
        info = (f"Subset => #Drugs={d_val}, #Individuals={i_val}\n"
                f"Correlation: {corr_val:.2f}\n"
                f"Drugs: {drugs}\n"
                f"Individuals: {individuals}\n")
        return info

    @app.callback(
        Output('click-data-cost', 'children'),
        [Input('heatmap_cost', 'clickData')]
    )
    def display_click_data_cost(clickData):
        if clickData is None:
            return "Click a cell in the Cost-Effectiveness Heatmap."
        point = clickData['points'][0]
        x_sub = int(point['x'])
        y_sub = int(point['y'])

        i_val = start_i + x_sub
        d_val = start_d + y_sub

        corr_val = corr_matrix_full[d_val - 1, i_val - 1]
        cost_val = 0.0
        if corr_val >= min_corr_global:
            cost_val = (corr_val / (i_val + d_val)) * 100.0

        drugs = drugs_used_dict_global.get(d_val, [])
        individuals = individuals_used_dict_global.get(i_val, [])
        info = (f"Subset => #Drugs={d_val}, #Individuals={i_val}\n"
                f"Correlation: {corr_val:.2f}\n"
                f"Cost-Effectiveness: {cost_val:.2f}\n"
                f"Drugs: {drugs}\n"
                f"Individuals: {individuals}\n")
        return info

    app.run_server(debug=True, use_reloader=False, port=8090)

###############################################################################
# Entry Point
###############################################################################
if __name__ == "__main__":
    main()
    run_dash_app()
