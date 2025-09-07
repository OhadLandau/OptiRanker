###############################################################################
# Trial-Optimizer — PB-MSE + Spearman version for *input data*
# ---------------------------------------------------------------------------
# • Scoring uses Position-Biased MSE (PB-MSE) and Spearman ρ everywhere.
# • All static figures are now saved as 600-dpi PNGs inside
#   “…/Publication Images/InputData/”.
# • Heat-maps keep white-border highlights (ρ ≥ threshold).
# • Dash dashboard shows Spearman ρ + cost-effectiveness and reveals
#   Drugs / Individuals on cell-click.
###############################################################################

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

# ── PNG helper & save directory (Input-data branch) ──────────────────────────
SAVE_DIR = os.path.join(os.getcwd(), "Publication Images", "InputData")
os.makedirs(SAVE_DIR, exist_ok=True)


def save_png(fig, fname):
    """Save *fig* as a 600-dpi PNG inside “…/Publication Images/InputData/”."""
    out = os.path.join(SAVE_DIR, fname)
    fig.savefig(out, dpi=600, bbox_inches="tight", transparent=False)
    print(f"[✓] saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fast Spearman ρ (if SciPy present) or fallback to NumPy
try:
    from scipy.stats import spearmanr as _scipy_spearmanr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def spearman_correlation(x, y):
    """Spearman rank-correlation."""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    if _HAS_SCIPY:
        rho, _ = _scipy_spearmanr(x, y, nan_policy="omit")
        return 0.0 if np.isnan(rho) else float(rho)

    # NumPy fallback (ties averaged)
    def _rank(a):
        a = np.asarray(a)
        sorter = np.argsort(a, kind="mergesort")
        inv = np.empty_like(sorter)
        inv[sorter] = np.arange(len(a))
        a_sorted = a[sorter]
        diffs = np.diff(a_sorted)
        ranks = inv.astype(float)
        if diffs.size:
            start = 0
            for k in range(len(a)):
                if k == len(a) - 1 or a_sorted[k] != a_sorted[k + 1]:
                    end = k
                    ranks[start:end + 1] = (start + end) / 2.0
                    start = k + 1
        return ranks + 1.0

    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# Position-Biased MSE helpers
BETA = 0.25  # changed by GUI input


def position_weight(rank):
    """Exponential discount; rank is 1-based."""
    return np.exp(-BETA * (rank - 1))


def pbmse(r_true, r_pred):
    """Position-Biased Mean-Squared Error for one ranked item."""
    return position_weight(r_true) * (r_true - r_pred) ** 2


###############################################################################
# KNN Imputation
###############################################################################
def knn_impute_data(df, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)


###############################################################################
# Data Pre-processing
###############################################################################
def preprocess_data(predictor_files, results_file=None,
                    is_ranked=False, no_results=False):
    """
    Read predictors/results, align on common drugs & individuals,
    K-NN-impute numeric values, convert to ranks (if needed).
    Returns data_dict and ranked DataFrames ready for PB-MSE scoring.
    """
    data_dict = {}
    predictor_dfs = []
    predictor_names = []

    for file_path in predictor_files:
        df = pd.read_csv(file_path, index_col=0)
        df.columns = df.columns.str.replace('.', '-', regex=False)
        if df.index[0] == "DRUG_NAME":
            df = df.drop(df.index[0])
        predictor_dfs.append(df)
        predictor_names.append(os.path.splitext(os.path.basename(file_path))[0])

    common_drugs = set(predictor_dfs[0].index)
    common_individuals = set(predictor_dfs[0].columns)
    for df in predictor_dfs[1:]:
        common_drugs &= set(df.index)
        common_individuals &= set(df.columns)

    df_results_ranked = None
    if not no_results and results_file:
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

    aligned_predictor_dfs = [df.loc[common_drugs, common_individuals]
                             for df in predictor_dfs]

    ranked_predictor_dfs = []
    if not is_ranked:
        for df in aligned_predictor_dfs:
            df_num = df.apply(pd.to_numeric, errors='coerce')
            df_imp = knn_impute_data(df_num)
            df_rank = df_imp.rank(axis=0, method='min',
                                  na_option='keep').astype(int)
            ranked_predictor_dfs.append(df_rank)
    else:
        for df in aligned_predictor_dfs:
            ranked_predictor_dfs.append(
                df.rank(axis=0, method='min', na_option='keep').astype(int))

    if df_results is not None:
        if not is_ranked:
            df_num = df_results.apply(pd.to_numeric, errors='coerce')
            df_imp = knn_impute_data(df_num)
            df_results_ranked = df_imp.rank(axis=0, method='min',
                                            na_option='keep').astype(int)
        else:
            df_results_ranked = df_results.rank(axis=0, method='min',
                                                na_option='keep').astype(int)

    for idx, df in enumerate(ranked_predictor_dfs):
        data_dict[f"Predictor{idx}"] = df.values.tolist()

    return (data_dict,
            common_drugs,
            common_individuals,
            df_results_ranked,
            ranked_predictor_dfs,
            predictor_names)


###############################################################################
# STD helpers
###############################################################################
def compute_drug_stds(predictor_dfs):
    rankings = [df.values for df in predictor_dfs]
    rankings = np.stack(rankings, axis=-1)
    drug_stds = np.std(rankings, axis=(1, 2))
    return pd.Series(drug_stds, index=predictor_dfs[0].index)


def compute_individual_stds(predictor_dfs):
    rankings = [df.values for df in predictor_dfs]
    rankings = np.stack(rankings, axis=-1)
    individual_stds = np.std(rankings, axis=(0, 2))
    return pd.Series(individual_stds, index=predictor_dfs[0].columns)


###############################################################################
# PB-MSE scoring helpers
###############################################################################
def calculate_scores_full(df_results_ranked, predictor_dfs):
    """Total PB-MSE of each predictor versus the results DataFrame."""
    scores = []
    for df in predictor_dfs:
        tot = 0.0
        for r in range(df.shape[0]):           # drugs
            for c in range(df.shape[1]):       # individuals
                r_true = int(df_results_ranked.iat[r, c])
                r_pred = int(df.iat[r, c])
                tot += pbmse(r_true, r_pred)
        scores.append(tot)
    return np.array(scores)


###############################################################################
# Correlation-based “no results” simulation helpers
###############################################################################
def RandomDataGenerator(num_individuals, num_drugs, num_predictors):
    """Generate synthetic predictors & a Gold Standard."""
    temp_sim_data = {}
    unique_ints = np.arange(1, num_drugs + 1)
    gold_standard = np.empty((num_drugs, num_individuals), dtype=int)
    for i in range(num_individuals):
        np.random.shuffle(unique_ints)
        gold_standard[:, i] = unique_ints
    temp_sim_data["GoldStandard"] = gold_standard.copy()
    temp_sim_data["Predictor0"] = gold_standard.copy()

    for k in range(1, num_predictors):
        degraded = gold_standard.copy()
        swaps = k
        for _ in range(swaps):
            ind = random.randint(0, num_individuals - 1)
            d1, d2 = random.sample(range(num_drugs), 2)
            degraded[d1, ind], degraded[d2, ind] = degraded[d2, ind], degraded[d1, ind]
        temp_sim_data[f"Predictor{k}"] = degraded
    return temp_sim_data, gold_standard


def run_project_simulation(num_individuals, num_drugs, num_predictors,
                           min_correlation, N,
                           common_drugs, common_individuals,
                           predictor_names, num_iterations):
    """Same workflow as original but with Spearman ρ."""
    all_subsets = []
    correlations_matrix = None

    sim_data, gold_standard = RandomDataGenerator(
        num_individuals, num_drugs, num_predictors)
    predictor_dfs_sim = []
    for key in sim_data:
        if key.startswith("Predictor"):
            predictor_dfs_sim.append(
                pd.DataFrame(sim_data[key],
                             index=common_drugs[:num_drugs],
                             columns=common_individuals[:num_individuals])
            )
    gold_df = pd.DataFrame(sim_data["GoldStandard"],
                           index=common_drugs[:num_drugs],
                           columns=common_individuals[:num_individuals])

    drug_stds = compute_drug_stds(predictor_dfs_sim)
    sorted_drugs = drug_stds.sort_values(ascending=False).index.tolist()
    ind_stds = compute_individual_stds(predictor_dfs_sim)
    sorted_inds = ind_stds.sort_values(ascending=False).index.tolist()

    correlations_matrix = np.zeros((num_drugs, num_individuals))

    for nd in range(1, num_drugs + 1):
        top_drugs = sorted_drugs[:nd]
        for ni in range(1, num_individuals + 1):
            top_inds = sorted_inds[:ni]
            gold_sub = gold_df.loc[top_drugs, top_inds]
            pred_subs = [df.loc[top_drugs, top_inds]
                         for df in predictor_dfs_sim]
            gold_rank = gold_sub.mean(axis=1).rank(method='min')
            corrs = []
            for dfp in pred_subs:
                p_rank = dfp.mean(axis=1).rank(method='min')
                cval = spearman_correlation(p_rank.values,
                                            gold_rank.values)
                corrs.append(cval)
            avg_corr = np.mean(corrs)
            correlations_matrix[nd-1, ni-1] = avg_corr
            all_subsets.append({
                "NumDrugs": nd,
                "NumIndividuals": ni,
                "Correlation": avg_corr,
                "Drugs": top_drugs,
                "Individuals": top_inds
            })

    subsets_df = pd.DataFrame(all_subsets)
    grouped = subsets_df.groupby(['NumDrugs', 'NumIndividuals']).agg({
        'Correlation': 'mean',
        'Drugs': 'first',
        'Individuals': 'first'
    }).reset_index()

    valid = grouped[grouped['Correlation'] >= min_correlation]
    if valid.empty:
        nd_first = ni_first = None
    else:
        valid = valid.sort_values(by=['NumDrugs', 'NumIndividuals'])
        top = valid.head(N)
        nd_first = top.iloc[0]['NumDrugs']
        ni_first = top.iloc[0]['NumIndividuals']

    return correlations_matrix, nd_first, ni_first, grouped


###############################################################################
# Optimisation when results are available  (Spearman + PB-MSE)
###############################################################################
def optimize_ranking_correlation(df_results_ranked, predictor_dfs,
                                 min_correlation, N):
    full_scores = calculate_scores_full(df_results_ranked, predictor_dfs)
    full_ranking = np.argsort(full_scores) + 1           # 1 = best

    num_drugs_total = len(df_results_ranked.index)
    num_inds_total = len(df_results_ranked.columns)

    drug_stds = compute_drug_stds(predictor_dfs)
    sorted_drugs = drug_stds.sort_values(ascending=False).index.tolist()
    sorted_inds = df_results_ranked.columns.tolist()

    correlations = np.zeros((num_drugs_total, num_inds_total))
    drugs_used_dict, inds_used_dict = {}, {}
    all_subsets = []

    for nd in range(1, num_drugs_total + 1):
        top_drugs = sorted_drugs[:nd]
        drugs_used_dict[nd] = top_drugs
        for ni in range(1, num_inds_total + 1):
            top_inds = sorted_inds[:ni]
            inds_used_dict[ni] = top_inds

            df_res_sub = df_results_ranked.loc[top_drugs, top_inds]
            pred_subs = [df.loc[top_drugs, top_inds] for df in predictor_dfs]
            sub_scores = calculate_scores_full(df_res_sub, pred_subs)
            sub_ranking = np.argsort(sub_scores) + 1
            cval = spearman_correlation(sub_ranking, full_ranking)
            correlations[nd-1, ni-1] = cval
            all_subsets.append({
                "NumDrugs": nd,
                "NumIndividuals": ni,
                "Correlation": cval,
                "Drugs": top_drugs,
                "Individuals": top_inds
            })

    subsets_df = pd.DataFrame(all_subsets)
    valid = subsets_df[subsets_df['Correlation'] >= min_correlation]
    if valid.empty:
        nd_first = ni_first = None
    else:
        valid = valid.sort_values(by=['NumDrugs', 'NumIndividuals'])
        top = valid.head(N)
        nd_first = top.iloc[0]['NumDrugs']
        ni_first = top.iloc[0]['NumIndividuals']

    return correlations, nd_first, ni_first, subsets_df, drugs_used_dict, inds_used_dict


###############################################################################
# Heat-maps & plots
###############################################################################
def plot_static_heatmap(correlations_matrix, min_corr,
                        n_drugs_first, n_inds_first,
                        subsets_df, N):
    if correlations_matrix is None or correlations_matrix.size == 0:
        return

    n_drugs, n_inds = correlations_matrix.shape
    labels = np.vectorize(lambda v: f"{v:.2f}")(correlations_matrix)

    fig, ax = plt.subplots(figsize=(min(20, n_inds/2+2), min(20, n_drugs/2+2)))
    sns.heatmap(correlations_matrix, cmap='Blues',
                annot=labels, fmt="",
                square=True, cbar=True,
                xticklabels=False, yticklabels=False, ax=ax)
    ax.set_title("Spearman ρ – full matrix")
    ax.invert_yaxis()
    plt.tight_layout()
    save_png(fig, "static_heatmap_inputdata_full.png")
    plt.show()

    if n_drugs_first is None or n_inds_first is None:
        return
    idr, iid = n_drugs_first-1, n_inds_first-1
    win = 10
    r0 = max(0, idr - win//2);   r1 = min(r0 + win, n_drugs)
    c0 = max(0, iid - win//2);   c1 = min(c0 + win, n_inds)
    zoom = correlations_matrix[r0:r1, c0:c1]
    zlab = np.vectorize(lambda v: f"{v:.2f}")(zoom)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(zoom, cmap='Blues',
                annot=zlab, fmt="",
                square=True, cbar=True,
                xticklabels=range(c0+1, c1+1),
                yticklabels=range(r0+1, r1+1), ax=ax)
    ax.set_xlabel("#Individuals")
    ax.set_ylabel("#Drugs")
    ax.set_title(f"Zoom 10×10 @ ({n_drugs_first} d, {n_inds_first} i)")
    ax.invert_yaxis()

    for r in range(zoom.shape[0]):
        for c in range(zoom.shape[1]):
            if zoom[r, c] >= min_corr:
                ax.add_patch(Rectangle((c, r), 1, 1, fill=False,
                                       edgecolor='white', linewidth=2))

    plt.tight_layout()
    save_png(fig, "static_heatmap_inputdata_zoom.png")
    plt.show()


def build_cost_dict(subsets_df):
    data_dict = {}
    if subsets_df is None or subsets_df.empty:
        return data_dict
    for _, row in subsets_df.iterrows():
        key = f"i{int(row['NumIndividuals'])}d{int(row['NumDrugs'])}"
        data_dict[key] = float(row['Correlation'])
    return data_dict


def plot_cost_effectiveness_heatmap(data_dict, min_corr):
    if not data_dict:
        return

    x_vals = sorted({int(lbl.split('i')[1].split('d')[0]) for lbl in data_dict})
    y_vals = sorted({int(lbl.split('d')[1]) for lbl in data_dict})
    cost = np.zeros((len(y_vals), len(x_vals)))

    for lbl, rho in data_dict.items():
        i = int(lbl.split('i')[1].split('d')[0])
        d = int(lbl.split('d')[1])
        r, c = y_vals.index(d), x_vals.index(i)
        cost[r, c] = (rho/(i+d))*100 if rho >= min_corr else 0

    if np.allclose(cost, 0):
        return

    vmax = cost.max()*1.05
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cost, cmap="YlOrRd",
                vmin=0, vmax=vmax,
                xticklabels=x_vals, yticklabels=y_vals, ax=ax)
    ax.set_xlabel("#Individuals")
    ax.set_ylabel("#Drugs")
    ax.set_title("Cost-Effectiveness (ρ / cost ×100)")
    ax.invert_yaxis()
    plt.tight_layout()
    save_png(fig, "cost_heatmap_inputdata.png")
    plt.show()


###############################################################################
# Permutation test
###############################################################################
def permutation_test(df_results_ranked, predictor_dfs, n_permutations=10):
    actual_scores = calculate_scores_full(df_results_ranked, predictor_dfs)
    perm_dist = np.zeros((n_permutations, len(predictor_dfs)))
    for p in range(n_permutations):
        perm_res = df_results_ranked.apply(np.random.permutation, axis=0)
        perm_dist[p] = calculate_scores_full(perm_res, predictor_dfs)
    p_vals = []
    for i in range(len(actual_scores)-1):
        for j in range(i+1, len(actual_scores)):
            diff = abs(actual_scores[i]-actual_scores[j])
            perm_diff = np.abs(perm_dist[:, i]-perm_dist[:, j])
            p_vals.append((i, j, np.mean(perm_diff >= diff)))
    return p_vals, actual_scores


###############################################################################
# Color helpers for predictor-consistent visuals
###############################################################################
def _make_palette(n: int):
    if n <= 8:
        return sns.color_palette("Set2", n)
    else:
        return sns.color_palette("tab20", n)


def get_predictor_color_map(predictor_labels):
    palette = _make_palette(len(predictor_labels))
    return {lbl: palette[i % len(palette)] for i, lbl in enumerate(predictor_labels)}


###############################################################################
# Simple bar / histogram / PCA plots
###############################################################################
def plot_bar_chart_with_significance(scores, predictor_labels, p_values):
    buffer = 0.1
    rng = max(scores) - min(scores)
    scaled = [1 - (buffer + (s - min(scores))/rng * (1-2*buffer)) if rng else 0.5
              for s in scores]

    color_map = get_predictor_color_map(predictor_labels)
    bar_colors = [color_map[lbl] for lbl in predictor_labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(predictor_labels, scaled, color=bar_colors)
    ax.set_ylim([0, 1])

    # Bigger, bold axis titles
    ax.set_xlabel("Predictors", fontsize=18, fontweight='bold')
    ax.set_ylabel("Score (Scaled WMSE)", fontsize=18, fontweight='bold')

    # Bigger, bold tick labels (algorithm names on x-axis + numbers on y-axis)
    ax.tick_params(axis='both', which='major', labelsize=14)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')

    plt.tight_layout()
    save_png(fig, "bar_chart_inputdata.png")
    plt.show()

def plot_histogram_for_individual_correlations(df_results_ranked, predictor_dfs, predictor_names):
    for idx, df_pred in enumerate(predictor_dfs):
        indiv_corrs = []
        for col in df_results_ranked.columns:
            indiv_corrs.append(
                spearman_correlation(df_results_ranked[col], df_pred[col]))
        overall = spearman_correlation(df_results_ranked.values.flatten(),
                                       df_pred.values.flatten())
        fig = plt.figure(figsize=(10, 4))
        plt.bar(df_results_ranked.columns, indiv_corrs, color='skyblue')
        plt.axhline(overall, color='red', ls='--', label=f'Overall ρ={overall:.2f}')
        plt.ylim([-1, 1])
        plt.xticks(rotation=45, ha='right')
        plt.title(f"{predictor_names[idx]} – Individual Spearman ρ")
        plt.legend()
        plt.tight_layout()
        save_png(fig, f"hist_indivcorr_{predictor_names[idx]}.png")
        plt.show()


def apply_pca_and_plot(df_results_ranked, predictor_dfs, predictor_names, has_results):
    from sklearn.decomposition import PCA

    predictor_color_map = get_predictor_color_map(predictor_names)
    results_color = (0.2, 0.2, 0.2)

    all_data, labels = [], []
    if has_results and df_results_ranked is not None:
        for c in df_results_ranked.columns:
            all_data.append(df_results_ranked[c].values)
            labels.append("Results")
    for idx, df in enumerate(predictor_dfs):
        for c in df.columns:
            all_data.append(df[c].values)
            labels.append(predictor_names[idx])
    all_data = np.array(all_data)
    if all_data.size == 0:
        return

    mean, std = all_data.mean(axis=0), all_data.std(axis=0)
    std[std == 0] = 1e-9
    all_std = (all_data - mean) / std

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(all_std)

    fig = plt.figure(figsize=(8, 6))
    for i, lbl in enumerate(labels):
        if lbl == "Results":
            color = results_color
        else:
            color = predictor_color_map.get(lbl, results_color)
        plt.scatter(pcs[i, 0], pcs[i, 1], color=color, s=60,
                    alpha=0.75, edgecolors='k')

    ax = plt.gca()
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=18, fontweight='bold')
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=18, fontweight='bold')

    # Bigger, bold tick labels (numbers on both axes)
    ax.tick_params(axis='both', which='major', labelsize=14)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')

    # Legend (kept, larger font; bold entries)
    handles = []
    for name in predictor_names:
        handles.append(plt.Line2D([], [], marker='o', ms=10, ls='',
                                  color=predictor_color_map[name],
                                  markeredgecolor='k', label=name))
    if has_results and df_results_ranked is not None:
        handles.append(plt.Line2D([], [], marker='o', ms=10, ls='',
                                  color=results_color,
                                  markeredgecolor='k', label='Results'))
    leg = plt.legend(handles=handles, fontsize=14)
    for text in leg.get_texts():
        text.set_fontweight('bold')

    plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    save_png(fig, "pca_inputdata.png")
    plt.show()


def plot_predictor_relationship_with_results_heatmap(pred_dfs, res_df, names):
    dfs, lbls = [], []
    if res_df is not None:
        dfs.append(res_df)
        lbls.append("Results")
    for i, df in enumerate(pred_dfs):
        dfs.append(df)
        lbls.append(names[i])
    n = len(dfs)
    if n == 0:
        return
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = spearman_correlation(dfs[i].values.flatten(),
                                             dfs[j].values.flatten())
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=lbls, yticklabels=lbls,
                cbar_kws={'label': 'Spearman ρ'})
    plt.title("Predictor/Results Correlation")
    plt.tight_layout()
    save_png(fig, "heatmap_predictor_relationships.png")
    plt.show()


###############################################################################
# PySimpleGUI input form  (added β field)
###############################################################################
def file_input_gui():
    layout = [
        [sg.Text('Select CSV files for each predictor (click "+" to add more):')],
        [sg.Column([[sg.Input(key='Predictor0'),
                     sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))]],
                   key='PredictorsColumn')],
        [sg.Button('+', key='AddPredictor')],
        [sg.Checkbox('No results file', key='NoResults', default=False,
                     enable_events=True)],
        [sg.Text('Select CSV file for results:', key='ResultsLabel'),
         sg.Input(key='Results'),
         sg.FileBrowse(file_types=(("CSV Files", "*.csv"),), key='ResultsBrowse')],
        [sg.Text('Minimum correlation (ρ):', key='MinCorrLabel'),
         sg.InputText('0.7', key='MinCorrelation')],
        [sg.Text('Number of subsets to find (N):', key='NumSubsetsLabel'),
         sg.InputText('1', key='NumSubsets')],
        [sg.Text('Top STD Individuals:', key='STDIndLabel', visible=False),
         sg.InputText('1', key='STDIndInput', visible=False)],
        [sg.Text('Top STD Drugs:', key='STDDrugLabel', visible=False),
         sg.InputText('1', key='STDDrugInput', visible=False)],
        [sg.Checkbox('Data is already ranked', key='Ranked', default=False)],
        [sg.Text('β (position-bias) [default 50]:'), sg.InputText('', key='BetaInput')],
        [sg.Button('Submit'), sg.Button('Cancel')]
    ]

    window = sg.Window('Data Input', layout, finalize=True)
    predictor_keys = ['Predictor0']

    while True:
        ev, val = window.read()
        if ev in (sg.WINDOW_CLOSED, 'Cancel'):
            window.close()
            return None
        if ev == 'AddPredictor':
            new_k = f"Predictor{len(predictor_keys)}"
            predictor_keys.append(new_k)
            window.extend_layout(window['PredictorsColumn'],
                                 [[sg.Input(key=new_k),
                                   sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))]])
            window.refresh()
        if ev == 'NoResults':
            show = not val['NoResults']
            for key in ('ResultsLabel', 'Results', 'ResultsBrowse',
                        'MinCorrLabel', 'MinCorrelation',
                        'NumSubsetsLabel', 'NumSubsets'):
                window[key].update(visible=show)
            for key in ('STDIndLabel', 'STDIndInput',
                        'STDDrugLabel', 'STDDrugInput'):
                window[key].update(visible=not show)
            window.refresh()
        if ev == 'Submit':
            pred_files = [val[k] for k in predictor_keys if val[k]]
            if not pred_files:
                sg.popup("Select at least one predictor file.")
                continue
            no_results = val['NoResults']
            is_ranked = val['Ranked']

            if no_results:
                try:
                    std_i = int(val['STDIndInput'])
                    std_d = int(val['STDDrugInput'])
                except ValueError:
                    sg.popup("Top STD fields must be integers.")
                    continue
                res_file = None
                min_corr = None
                n_sub = None
                n_iter = None
            else:
                res_file = val['Results']
                if not res_file:
                    sg.popup("Select a results file or choose 'No results'.")
                    continue
                try:
                    min_corr = float(val['MinCorrelation'])
                    n_sub = int(val['NumSubsets'])
                except ValueError:
                    sg.popup("Enter valid numeric min-corr / subset count.")
                    continue
                std_i = std_d = None
                n_iter = None

            beta_in = val['BetaInput'].strip()
            try:
                beta_val = float(beta_in) if beta_in else 0.25
            except ValueError:
                sg.popup("β must be numeric.")
                continue

            window.close()
            return (pred_files, res_file, is_ranked, no_results,
                    min_corr, n_sub, n_iter, std_i, std_d, beta_val)


###############################################################################
# Globals populated later
###############################################################################
drugs_used_dict_global = {}
individuals_used_dict_global = {}
data_dict_global = {}
min_corr_global = 0.7


###############################################################################
# Main
###############################################################################
def main():
    global BETA, drugs_used_dict_global, individuals_used_dict_global
    global data_dict_global, min_corr_global

    res = file_input_gui()
    if res is None:
        return
    (pred_files, results_file, is_ranked, no_results,
     min_corr, n_subsets, n_iter, top_i, top_d, beta_val) = res

    BETA = beta_val
    print(f"β set to {BETA}")

    try:
        (data_dict, common_drugs, common_individuals,
         df_results_ranked, ranked_pred_dfs, pred_names) = preprocess_data(
            pred_files, results_file, is_ranked, no_results)

        if no_results:
            std_drugs = compute_drug_stds(ranked_pred_dfs).sort_values(ascending=False)
            std_inds = compute_individual_stds(ranked_pred_dfs).sort_values(ascending=False)
            chosen_drugs = list(std_drugs.index[:top_d])
            chosen_inds = list(std_inds.index[:top_i])
            print("\n########## Highest STD ##########")
            print(f"Top {top_d} Drugs:", chosen_drugs)
            print(f"Top {top_i} Individuals:", chosen_inds)
            print("#################################")

            (corr_mat, nd_first, ni_first, subsets_df) = run_project_simulation(
                len(chosen_inds), len(chosen_drugs),
                len(ranked_pred_dfs), 0.0, 1,
                common_drugs, common_individuals,
                pred_names, 1)
            plot_static_heatmap(corr_mat, 0.0, nd_first, ni_first, subsets_df, 1)

        else:
            min_corr_global = min_corr
            p_vals, full_scores = permutation_test(df_results_ranked, ranked_pred_dfs)
            plot_bar_chart_with_significance(full_scores, pred_names, p_vals)
            plot_histogram_for_individual_correlations(df_results_ranked,
                                                       ranked_pred_dfs, pred_names)
            apply_pca_and_plot(df_results_ranked, ranked_pred_dfs, pred_names, True)
            plot_predictor_relationship_with_results_heatmap(ranked_pred_dfs,
                                                             df_results_ranked, pred_names)

            (corrs, nd_first, ni_first, subsets_df,
             drugs_dict, inds_dict) = optimize_ranking_correlation(
                df_results_ranked, ranked_pred_dfs,
                min_corr, n_subsets)

            drugs_used_dict_global = drugs_dict
            individuals_used_dict_global = inds_dict

            plot_static_heatmap(corrs, min_corr, nd_first, ni_first, subsets_df, n_subsets)
            data_dict_global = build_cost_dict(subsets_df)
            plot_cost_effectiveness_heatmap(data_dict_global, min_corr)

    except Exception as e:
        print("Error:", e)


###############################################################################
# Dash app   — Spearman ρ  +  Cost-effectiveness
###############################################################################
def run_dash_app():
    subset_labels = list(data_dict_global.keys())
    if not subset_labels:
        print("No data for Dash.")
        return

    max_i = max(int(lbl.split('i')[1].split('d')[0]) for lbl in subset_labels)
    max_d = max(int(lbl.split('d')[1]) for lbl in subset_labels)
    corr_full = np.zeros((max_d, max_i))
    for lbl, val in data_dict_global.items():
        i = int(lbl.split('i')[1].split('d')[0])
        d = int(lbl.split('d')[1])
        corr_full[d - 1, i - 1] = val

    best_key = max(data_dict_global, key=data_dict_global.get)
    best_i   = int(best_key.split('i')[1].split('d')[0])
    best_d   = int(best_key.split('d')[1])
    win      = 10
    s_i = max(1, best_i - win // 2)
    e_i = min(max_i, s_i + win - 1)
    s_d = max(1, best_d - win // 2)
    e_d = min(max_d, s_d + win - 1)
    corr_sub = corr_full[s_d - 1:e_d, s_i - 1:e_i]

    cost_sub = np.zeros_like(corr_sub)
    for r in range(corr_sub.shape[0]):
        for c in range(corr_sub.shape[1]):
            rho = corr_sub[r, c]
            cost_sub[r, c] = (rho / ((s_i + c) + (s_d + r)) * 100
                              if rho >= min_corr_global else 0)

    x_lbls = [str(i) for i in range(s_i, e_i + 1)]
    y_lbls = [str(d) for d in range(s_d, e_d + 1)]

    eps = 1e-6
    cs_corr = [
        [0.00, '#ffffff'],
        [0.60, '#cce5ff'],
        [min_corr_global, '#003c8f'],
        [min_corr_global + eps, '#ffd6d6'],
        [1.00, '#b30000']
    ]

    def build_ann_and_shapes(z_mat, show_numbers):
        ann, shapes = [], []
        for r in range(z_mat.shape[0]):
            for c in range(z_mat.shape[1]):
                if corr_sub[r, c] >= min_corr_global:
                    if show_numbers:
                        txt = f"{z_mat[r, c]:.2f}" if show_numbers == 'corr' \
                              else f"{z_mat[r, c]:.1f}"
                        ann.append(
                            dict(text=txt, x=c, y=r, showarrow=False,
                                 font=dict(
                                     color=('white' if corr_sub[r, c] >= 0.75 else 'black'),
                                     size=10, family='Arial Black'))
                        )
                    shapes.append(
                        dict(type='rect',
                             x0=c-0.5, x1=c+0.5, y0=r-0.5, y1=r+0.5,
                             line=dict(color='white', width=2))
                    )
        return ann, shapes

    ann_corr, shapes_corr = build_ann_and_shapes(corr_sub, 'corr')
    heat_corr = go.Heatmap(z=corr_sub, zmin=0, zmax=1,
                           colorscale=cs_corr, hoverinfo='x+y+z',
                           x=list(range(len(x_lbls))), y=list(range(len(y_lbls))))

    fig_corr = go.Figure(data=[heat_corr],
        layout=go.Layout(
            title="Spearman ρ",
            annotations=ann_corr,
            shapes=shapes_corr,
            xaxis=dict(title="#Individuals",
                       tickvals=list(range(len(x_lbls))), ticktext=x_lbls,
                       tickangle=-45),
            yaxis=dict(title="#Drugs",
                       tickvals=list(range(len(y_lbls))), ticktext=y_lbls))
    )

    ann_cost, shapes_cost = build_ann_and_shapes(cost_sub, 'cost')
    heat_cost = go.Heatmap(z=cost_sub, zmin=0, zmax=max(cost_sub.max(), 1),
                           colorscale='YlOrRd', hoverinfo='x+y+z',
                           x=list(range(len(x_lbls))), y=list(range(len(y_lbls))))

    fig_cost = go.Figure(data=[heat_cost],
        layout=go.Layout(
            title="Cost-Effectiveness (ρ / cost ×100)",
            annotations=ann_cost,
            shapes=shapes_cost,
            xaxis=dict(title="#Individuals",
                       tickvals=list(range(len(x_lbls))), ticktext=x_lbls,
                       tickangle=-45),
            yaxis=dict(title="#Drugs",
                       tickvals=list(range(len(y_lbls))), ticktext=y_lbls))
    )

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='heat_corr', figure=fig_corr,
                      style={'display': 'inline-block', 'width': '48%'}),
            dcc.Graph(id='heat_cost', figure=fig_cost,
                      style={'display': 'inline-block', 'width': '48%'}),
        ]),
        html.Div(id='click-data', style={'whiteSpace': 'pre-line',
                                         'padding': '20px',
                                         'fontSize': '16px'})
    ])

    @app.callback(Output('click-data', 'children'),
                  [Input('heat_corr', 'clickData'),
                   Input('heat_cost', 'clickData')])
    def display_click(corr_click, cost_click):
        ctx = dash.callback_context
        if not ctx.triggered:
            return "Click a cell!"
        clickData = ctx.triggered[0]['value']
        if not clickData:
            return "Click a cell!"
        r = clickData['points'][0]['y']
        c = clickData['points'][0]['x']
        i_val = s_i + c
        d_val = s_d + r
        rho  = corr_sub[r, c]
        cost = cost_sub[r, c]
        drugs = drugs_used_dict_global.get(d_val, [])
        inds  = individuals_used_dict_global.get(i_val, [])

        return (f"#Drugs = {d_val}, #Individuals = {i_val}\n"
                f"Spearman ρ = {rho:.2f}\n"
                f"Cost-Effectiveness = {cost:.2f}\n"
                f"Drugs: {drugs}\n"
                f"Individuals: {inds}")

    app.run_server(debug=False, use_reloader=False, port=8080)


###############################################################################
# Entry
###############################################################################
if __name__ == "__main__":
    main()
    run_dash_app()
