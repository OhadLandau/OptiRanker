import warnings
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice",
                        category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                        category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide",
                        category=RuntimeWarning)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
import numpy as np
import random
import os
import PySimpleGUI as sg
from itertools import chain
from sklearn.preprocessing import MinMaxScaler
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import csv




_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PUB_DIR  = os.path.join(_BASE_DIR, "Publication Images")
os.makedirs(_PUB_DIR, exist_ok=True)

def save_png(fig, filename: str):
    path = os.path.join(_PUB_DIR, filename)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"[saved] {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Optional SciPy for faster Spearman ρ (falls back to NumPy if missing)
try:
    from scipy.stats import spearmanr as _scipy_spearmanr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
# ─────────────────────────────────────────────────────────────────────────────
# Global, user-set β (initial default = 50; updated in main())
BETA = 0.25


def position_weight(rank: int) -> float:
    """Exponential discount; rank is 1-based (top item rank == 1)."""
    return np.exp(-BETA * (rank - 1))


def pbmse(r_true: int, r_pred: int) -> float:
    """Position-Biased Mean-Squared Error for a single item."""
    return position_weight(r_true) * (r_true - r_pred) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Spearman rank-correlation
def spearman_correlation(x, y) -> float:
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

###############################################################################
# GLOBALS  (populated after main runs)
###############################################################################
data_dict_global = {}
scores_byD_high_std_global = {}
scores_byI_high_std_global = {}
min_corr_global = 0.0
drugs_used_dict_global = {}
individuals_used_dict_global = {}

###############################################################################
# HELPER to Compare Arbitrary vs. STD-based for Each Axis
###############################################################################
def compare_selection_methods_for_run_project(
    num_individual, num_drugs,
    data, list_of_stds, individual_stds,
    subset_correlation_func
):
    """Compares avg ρ for 2-drug / 2-individual subsets (arbitrary vs STD)."""
    c_arbitrary_drugs = subset_correlation_func(
        num_individual, 2, data, list_of_stds, individual_stds,
        select_high_std_drugs=False, select_high_std_individuals=False
    )
    c_std_drugs = subset_correlation_func(
        num_individual, 2, data, list_of_stds, individual_stds,
        select_high_std_drugs=True,  select_high_std_individuals=False
    )
    c_arbitrary_inds = subset_correlation_func(
        2, num_drugs, data, list_of_stds, individual_stds,
        select_high_std_drugs=False, select_high_std_individuals=False
    )
    c_std_inds = subset_correlation_func(
        2, num_drugs, data, list_of_stds, individual_stds,
        select_high_std_drugs=False, select_high_std_individuals=True
    )
    use_std_drugs = c_std_drugs >= c_arbitrary_drugs
    use_std_inds  = c_std_inds  >= c_arbitrary_inds
    return (use_std_drugs, use_std_inds,
            c_arbitrary_drugs, c_std_drugs,
            c_arbitrary_inds,  c_std_inds)

###############################################################################
# MAIN
###############################################################################
def main():
    global BETA

    num_iterations = int(
        input("How many simulations would you like to run? (1 = single) : ")
    )

    print("Please enter three dimensions:")
    num_individual = int(input("  • rows (individuals)           : "))
    num_predictors = int(input("  • key (3rd dimension – predictors): "))
    num_drugs      = int(input("  • columns (drugs)              : "))

    if num_individual != num_drugs:
        print(f"NOTE: The dataset is asymmetrical ({num_individual} × {num_drugs}).")

    corr = float(input("Enter the minimum acceptable Spearman ρ : "))

    # ── NEW  β prompt ────────────────────────────────────────────────────────
    beta_str = input(
        "Enter β (position-bias coefficient, default 50 – smaller β → gentler,"
        " larger β → harsher) : "
    ).strip()
    if beta_str:
        try:
            BETA = float(beta_str)
        except ValueError:
            print("Invalid β – using default 50")
            BETA = 0.25
    else:
        BETA = 0.25
    print(f"β set to {BETA}")

    if num_iterations > 1:
        run_multiple_simulations(
            num_individual, num_drugs, num_predictors,
            corr, None, None, num_iterations
        )
    else:
        print("\n--- Single simulation run ---")
        results, drugs_used_dict, individuals_used_dict = run_project(
            num_individual, num_drugs, num_predictors, corr,
            None, None, True, True
        )

        (
            best_subsetDict,
            scores_byD_arbitrary, scores_byI_arbitrary,
            corrByD_arbitrary,   corrByI_arbitrary,
            best_predictor_scores,
            list_of_stds, individual_stds,
            scores_byD_high_std, corrByD_high_std,
            scores_byI_high_std, corrByI_high_std
        ) = results

        plot_combined_scores([best_predictor_scores], best_predictor_scores)
        plot_combined_graphs(
            scores_byD_arbitrary, corrByD_arbitrary,
            scores_byI_arbitrary, corrByI_arbitrary,
            scores_byD_high_std,  corrByD_high_std,
            scores_byI_high_std,  corrByI_high_std,
            list_of_stds, individual_stds
        )
        plot_static_heatmap(best_subsetDict,
                            title="Single Simulation Heatmap of Subsets (Spearman ρ)",
                            min_corr=corr)

        global data_dict_global, scores_byD_high_std_global, scores_byI_high_std_global
        global min_corr_global, drugs_used_dict_global, individuals_used_dict_global

        data_dict_global             = best_subsetDict
        scores_byD_high_std_global   = scores_byD_high_std
        scores_byI_high_std_global   = scores_byI_high_std
        min_corr_global              = corr
        drugs_used_dict_global       = drugs_used_dict
        individuals_used_dict_global = individuals_used_dict

    # After single or multiple runs:
    plot_cost_effectiveness_heatmap(data_dict_global, min_corr_global)
    log_top_correlation_subsets(
        data_dict_global, min_corr_global,
        drugs_used_dict_global, individuals_used_dict_global, topN=10
    )
    log_top_cost_subsets(
        data_dict_global, min_corr_global,
        drugs_used_dict_global, individuals_used_dict_global, topN=10
    )
    plot_zoomed_in_correlation_heatmap(data_dict_global, min_corr_global, topN=10)
    plot_zoomed_in_cost_heatmap(data_dict_global, min_corr_global, topN=10)
    run_dash_app(port=8060)

###############################################################################
# FILE/IO UTILITIES
###############################################################################
def convert_file_to_csv(file_path):
    file_extension = os.path.splitext(file_path)[1]
    delimiter = '\t' if file_extension == '.txt' else ','
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip().split(delimiter) for line in lines]


def read_csv(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.values.tolist()


def clean_data(data, num_predictors):
    cleaned_data = {}
    for i in range(num_predictors):
        predictor_key = f"Predictor{i}"
        cleaned_data[predictor_key] = [[v for v in row if v != 0]
                                       for row in data[predictor_key]]
    return cleaned_data


def get_ranking(row):
    return (len(row) - np.argsort(np.argsort(row))).tolist()


def rank_data(data, num_predictors, num_drugs):
    ranked_data = {}
    for predictor_key, predictor_data in data.items():
        headers = predictor_data[0]
        ranked_predictor_data = [headers]
        for row_idx, row in enumerate(predictor_data):
            if row_idx == 0:
                continue
            ranked_predictor_data.append(get_ranking(row))
        ranked_data[predictor_key] = ranked_predictor_data
    return ranked_data


def export_ranked_data(ranked_data, num_predictors):
    for i in range(num_predictors):
        predictor_key = f"Predictor{i}"
        ranked_rows = ranked_data[predictor_key]
        with open(f"ranked_{predictor_key}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"Drug {j + 1}" for j in range(len(ranked_rows[0]))])
            for row in ranked_rows:
                writer.writerow(row)

###############################################################################
# BASIC HELPER FUNCTIONS
###############################################################################
def rank_integers(lst):
    eps = 1e-9
    noisy = [val + random.uniform(0, eps) for val in lst]
    sorted_vals = sorted(noisy, reverse=True)
    rank_map, cur = {}, 1
    for val in sorted_vals:
        if val not in rank_map:
            rank_map[val] = cur
            cur += 1
    return [rank_map[v] for v in noisy]


def normalize_and_invert(scores, margin: float = 0.05):
    """
    Min‑max‑scale ➜ squeeze into (margin … 1‑margin) ➜ invert.
    • Prevents a predictor ever landing on the exact 0 or 1 extremes,
      so P₁ isn’t a solid wall and P₂₀ still has a visible bar.
    • Keeps the relative distances identical, only shrunken slightly.
    • 'margin' (0 ⇢ 0.5) controls how close we allow values to hit
      the edges; default 0.05 → final range [0.05 … 0.95] before
      inversion, i.e. [0.05 … 0.95] after inversion as well.
    """
    if not scores:
        return []

    arr = np.asarray(scores, dtype=float).reshape(-1, 1)

    # Standard 0‑to‑1 scaling
    scaled = MinMaxScaler().fit_transform(arr).flatten()

    # scaled==0   →  margin
    # scaled==1   → 1‑margin
    if margin > 0:
        scaled = margin + (1.0 - 2*margin) * scaled

    # Invert so “larger = better”
    return (1.0 - scaled).tolist()


###############################################################################
# SCORING & RANDOM DATA GENERATION
###############################################################################
def drug_standardDeviation(d, data, num_individuals, num_predictors):
    return sum(np.std([data[p][ind][d] for p in data])
               for ind in range(num_individuals))


def RandomDataGenerator(num_individuals, num_drugs, num_predictors, gold_standard=None):
    """
    Predictor 0 = Gold Standard.  For predictor k (1…num_predictors-1):
      • perform exactly k random swaps,
      • keep only if its total PB-MSE is strictly worse than predictor k-1.
    Guarantees monotone deterioration with fixed swap counts.
    """
    temp_sim_data = {}

    # ── Gold Standard ────────────────────────────────────────────────────────
    if gold_standard is None:
        unique_ints = np.arange(1, num_drugs + 1)
        gold_standard = np.empty((num_individuals, num_drugs), dtype=int)
        for i in range(num_individuals):
            np.random.shuffle(unique_ints)
            gold_standard[i] = unique_ints
    else:
        gold_standard = np.array(gold_standard)
        if gold_standard.ndim == 3:
            gold_standard = gold_standard[0]

    temp_sim_data["Predictor 0"] = gold_standard
    temp_sim_data["Gold Standard"] = gold_standard

    # helper: total PB-MSE fast
    def total_pbmse(matrix):
        tot = 0.0
        for ind in range(num_individuals):
            for d in range(num_drugs):
                tot += pbmse(gold_standard[ind][d], matrix[ind][d])
        return tot

    prev_err = 0.0  # predictor 0 error is zero

    # ── Generate degraded predictors ────────────────────────────────────────
    for k in range(1, num_predictors):
        while True:
            degraded = np.copy(temp_sim_data[f"Predictor {k-1}"])
            for _ in range(k):                            # exactly k swaps
                ind = random.randint(0, num_individuals - 1)
                d1, d2 = random.sample(range(num_drugs), 2)
                degraded[ind][d1], degraded[ind][d2] = degraded[ind][d2], degraded[ind][d1]
            err = total_pbmse(degraded)
            if err > prev_err:                            # keep only if worse
                prev_err = err
                temp_sim_data[f"Predictor {k}"] = degraded
                break

    return temp_sim_data, {"Test Results": gold_standard}


def listTheStds(data, num_individuals, num_predictors, num_drugs):
    return [drug_standardDeviation(i, data, num_individuals, num_predictors)
            for i in range(num_drugs)]


def drug_selector(d, list_of_stds):
    return sorted(range(len(list_of_stds)),
                  key=lambda i: list_of_stds[i], reverse=True)[:d]


def individual_standardDeviation(data, num_individuals, *_):
    stds = []
    for individual in range(num_individuals):
        values = [data[predictor][individual] for predictor in data.keys()]
        stds.append(np.sum(np.std(values, axis=0)))
    return stds


def individual_selector(i, list_of_stds):
    return sorted(range(len(list_of_stds)),
                  key=lambda idx: list_of_stds[idx], reverse=True)[:i]

###############################################################################
# subset_correlation (PB-MSE + Spearman)
###############################################################################
def subset_correlation_func(i_count, d_count, data, list_of_stds, individual_stds,
                            select_high_std_drugs, select_high_std_individuals):
    """
    1) Score predictors on a subset using PB-MSE → min-max-invert → ranks.
    2) Compute Spearman ρ between subset ranks and full-data ranks.
    """
    def score_predictors(dset, drug_filter, indiv_filter):
        predictor_keys = list(dset.keys())[:-1]
        scores = []
        for predictor in predictor_keys:
            total = 0.0
            for ind in indiv_filter:
                for d in drug_filter:
                    total += pbmse(dset["Gold Standard"][ind][d],
                                   dset[predictor][ind][d])
            scores.append(total)
        return normalize_and_invert(scores)

    num_individual = len(data["Gold Standard"])
    num_drugs      = len(data["Gold Standard"][0])

    drug_filt = (drug_selector(d_count, list_of_stds)
                 if select_high_std_drugs else list(range(d_count)))
    indiv_filt = (individual_selector(i_count, individual_stds)
                  if select_high_std_individuals else list(range(i_count)))

    sub_scores  = score_predictors(data, drug_filt, indiv_filt)
    sub_ranks   = rank_integers(sub_scores)
    full_scores = score_predictors(data,
                                   list(range(num_drugs)),
                                   list(range(num_individual)))
    full_ranks  = rank_integers(full_scores)
    rho = spearman_correlation(sub_ranks, full_ranks)
    return 1.0 if (i_count == num_individual and d_count == num_drugs) else rho

###############################################################################
# CORE: run_project
###############################################################################
def run_project(num_individual, num_drugs, num_predictors, corr, _distance_unused,
                data, select_high_std_drugs=False, select_high_std_individuals=False):

    simulated_Data, _ = RandomDataGenerator(num_individual, num_drugs,
                                            num_predictors, data)
    list_of_stds = listTheStds(simulated_Data, num_individual,
                               num_predictors, num_drugs)
    if data is None:
        data = simulated_Data
    individual_stds = individual_standardDeviation(data, num_individual)

    (use_std_drugs, use_std_inds,
     c_arb_d, c_std_d, c_arb_i, c_std_i) = compare_selection_methods_for_run_project(
        num_individual, num_drugs, data, list_of_stds, individual_stds,
        subset_correlation_func
    )
    print(f"Drug selection method      : {'High STD' if use_std_drugs else 'Arbitrary'} "
          f"(ρ STD={c_std_d:.3f}, Arb={c_arb_d:.3f})")
    print(f"Individual selection method: {'High STD' if use_std_inds  else 'Arbitrary'} "
          f"(ρ STD={c_std_i:.3f}, Arb={c_arb_i:.3f})")

    select_high_std_drugs       = use_std_drugs
    select_high_std_individuals = use_std_inds

    def score_predictors(dset, drug_filter, indiv_filter):
        predictor_keys = list(dset.keys())[:-1]
        scores = []
        for predictor in predictor_keys:
            total = 0.0
            for ind in indiv_filter:
                for d in drug_filter:
                    total += pbmse(dset["Gold Standard"][ind][d],
                                   dset[predictor][ind][d])
            scores.append(total)
        return normalize_and_invert(scores)

    def subset_correlation(i_cnt, d_cnt):
        drug_f = (drug_selector(d_cnt, list_of_stds)
                  if select_high_std_drugs else list(range(d_cnt)))
        ind_f  = (individual_selector(i_cnt, individual_stds)
                  if select_high_std_individuals else list(range(i_cnt)))
        sub = score_predictors(data, drug_f, ind_f)
        subR = rank_integers(sub)
        full = score_predictors(data, list(range(num_drugs)),
                                list(range(num_individual)))
        fullR = rank_integers(full)
        return 1.0 if (i_cnt == num_individual and d_cnt == num_drugs) else \
               spearman_correlation(subR, fullR)

    best_subsetDict = {}
    drugs_used_dict = {}
    individuals_used_dict = {}

    for i_ in range(1, num_individual + 1):
        for d_ in range(1, num_drugs + 1):
            best_subsetDict[f"i{i_}d{d_}"] = subset_correlation(i_, d_)
            drugs_used_dict[d_] = (drug_selector(d_, list_of_stds)
                                   if select_high_std_drugs else list(range(d_)))
            individuals_used_dict[i_] = (individual_selector(i_, individual_stds)
                                         if select_high_std_individuals else list(range(i_)))

    # Helper for plot dictionaries
    def build_axis_dict(axis_len, is_drug_axis, high_std):
        scores_dict, corr_list = {}, []
        full = score_predictors(data, list(range(num_drugs)),
                                list(range(num_individual)))
        fullR = rank_integers(full)
        for n in range(1, axis_len + 1):
            if is_drug_axis:
                drg = (drug_selector(n, list_of_stds) if high_std else list(range(n)))
                ind = list(range(num_individual))
            else:
                drg = list(range(num_drugs))
                ind = (individual_selector(n, individual_stds) if high_std else list(range(n)))
            sub = score_predictors(data, drg, ind)
            scores_dict[str(n)] = sub
            corr_list.append(abs(spearman_correlation(rank_integers(sub), fullR)))
        return scores_dict, corr_list

    scores_byD_arbitrary, corrByD_arbitrary = build_axis_dict(num_drugs, True,  False)
    scores_byI_arbitrary, corrByI_arbitrary = build_axis_dict(num_individual, False, False)
    scores_byD_high_std, corrByD_high_std   = build_axis_dict(num_drugs, True,  True)
    scores_byI_high_std, corrByI_high_std   = build_axis_dict(num_individual, False, True)
    best_predictor_scores = score_predictors(
        data, list(range(num_drugs)), list(range(num_individual))
    )

    results = (
        best_subsetDict,
        scores_byD_arbitrary, scores_byI_arbitrary,
        corrByD_arbitrary,   corrByI_arbitrary,
        best_predictor_scores,
        list_of_stds, individual_stds,
        scores_byD_high_std, corrByD_high_std,
        scores_byI_high_std, corrByI_high_std
    )
    return results, drugs_used_dict, individuals_used_dict

###############################################################################
# run_multiple_simulations
###############################################################################
def run_multiple_simulations(num_individual, num_drugs, num_predictors,
                             corr, _distance_unused, data, num_iterations):
    all_best_predictor_scores = []
    all_scores_byD_arbitrary  = []
    all_corrByD_arbitrary     = []
    all_scores_byI_arbitrary  = []
    all_corrByI_arbitrary     = []
    all_scores_byD_high_std   = []
    all_corrByD_high_std      = []
    all_scores_byI_high_std   = []
    all_corrByI_high_std      = []
    all_best_subsetDict       = []

    global data_dict_global, scores_byD_high_std_global, scores_byI_high_std_global
    global min_corr_global, drugs_used_dict_global, individuals_used_dict_global

    for iteration_idx in range(num_iterations):
        print(f"\n--- Simulation iteration {iteration_idx + 1} of {num_iterations} ---")

        results, d_used, i_used = run_project(
            num_individual, num_drugs, num_predictors,
            corr, None, data, True, True
        )
        (
            best_subsetDict,
            scores_byD_arbitrary, scores_byI_arbitrary,
            corrByD_arbitrary,   corrByI_arbitrary,
            best_predictor_scores,
            list_of_stds, individual_stds,
            scores_byD_high_std, corrByD_high_std,
            scores_byI_high_std, corrByI_high_std
        ) = results

        drugs_used_dict_global       = d_used
        individuals_used_dict_global = i_used

        all_best_predictor_scores.append(best_predictor_scores)
        all_scores_byD_arbitrary.append(scores_byD_arbitrary)
        all_corrByD_arbitrary.append(corrByD_arbitrary)
        all_scores_byI_arbitrary.append(scores_byI_arbitrary)
        all_corrByI_arbitrary.append(corrByI_arbitrary)
        all_scores_byD_high_std.append(scores_byD_high_std)
        all_corrByD_high_std.append(corrByD_high_std)
        all_scores_byI_high_std.append(scores_byI_high_std)
        all_corrByI_high_std.append(corrByI_high_std)
        all_best_subsetDict.append(best_subsetDict)

    averaged_best_predictor_scores = np.mean(all_best_predictor_scores, axis=0).tolist()

    def average_dicts(dicts):
        all_keys = set(chain.from_iterable(d.keys() for d in dicts))
        avg_dict = {}
        for key in all_keys:
            matching = [d[key] for d in dicts if key in d]
            arr = np.array(matching, dtype=float)
            avg_dict[key] = (np.mean(arr, axis=0).tolist() if arr.ndim > 1
                             else float(np.mean(arr)))
        return avg_dict

    averaged_scores_byD_arbitrary = average_dicts(all_scores_byD_arbitrary)
    averaged_corrByD_arbitrary    = np.mean(all_corrByD_arbitrary, axis=0).tolist()
    averaged_scores_byI_arbitrary = average_dicts(all_scores_byI_arbitrary)
    averaged_corrByI_arbitrary    = np.mean(all_corrByI_arbitrary, axis=0).tolist()
    averaged_scores_byD_high_std  = average_dicts(all_scores_byD_high_std)
    averaged_corrByD_high_std     = np.mean(all_corrByD_high_std, axis=0).tolist()
    averaged_scores_byI_high_std  = average_dicts(all_scores_byI_high_std)
    averaged_corrByI_high_std     = np.mean(all_corrByI_high_std, axis=0).tolist()
    averaged_best_subsetDict      = average_dicts(all_best_subsetDict)

    plot_combined_scores(all_best_predictor_scores, averaged_best_predictor_scores)
    plot_combined_graphs(
        averaged_scores_byD_arbitrary, averaged_corrByD_arbitrary,
        averaged_scores_byI_arbitrary, averaged_corrByI_arbitrary,
        averaged_scores_byD_high_std, averaged_corrByD_high_std,
        averaged_scores_byI_high_std, averaged_corrByI_high_std,
        list_of_stds, individual_stds
    )
    plot_static_heatmap(averaged_best_subsetDict,
                        title="Averaged Heatmap of Subsets (Spearman ρ)",
                        min_corr=corr)

    data_dict_global           = averaged_best_subsetDict
    scores_byD_high_std_global = averaged_scores_byD_high_std
    scores_byI_high_std_global = averaged_scores_byI_high_std
    min_corr_global            = corr

###############################################################################
# PLOTTING
###############################################################################
def plot_combined_scores(all_best_predictor_scores, averaged_best_predictor_scores):
    fig, axs = plt.subplots(2, 1, figsize=(18, 14))

    # ── bar chart of averaged scores ─────────────────────────────────────────
    ax1 = axs[0]
    avg_scores = averaged_best_predictor_scores
    ax1.bar(range(len(avg_scores)), avg_scores, color=sns.color_palette("Set2", len(avg_scores)))
    ax1.set_xlabel("Predictors", fontsize=20, fontweight='bold')
    ax1.set_ylabel("Avg Score (1 = best)", fontsize=20, fontweight='bold')
    ax1.set_title("Predictor Scores", fontsize=24, fontweight='bold')
    ax1.set_xticks(range(len(avg_scores)))
    ax1.set_xticklabels([f"P{i + 1}" for i in range(len(avg_scores))], fontsize=16)

    # ── box + jitter across simulations ─────────────────────────────────────
    ax2 = axs[1]
    data_box = []
    if all_best_predictor_scores:
        num_preds = len(all_best_predictor_scores[0])
        for i in range(num_preds):
            predictor_scores = [sim[i] for sim in all_best_predictor_scores]
            data_box.append(predictor_scores)
        box = ax2.boxplot(data_box, patch_artist=True, showmeans=True, meanline=True)
        palette = sns.color_palette("Set2", len(avg_scores))
        for i, patch in enumerate(box['boxes']):
            patch.set_facecolor('none')
            patch.set_edgecolor(palette[i])
            patch.set_linewidth(2)
            for elem in ['whiskers', 'caps', 'medians', 'means']:
                plt.setp(box[elem][2*i:2*(i+1)], color=palette[i], linewidth=2)
            if i < len(box['fliers']):
                plt.setp(box['fliers'][i], markerfacecolor=palette[i], markeredgecolor=palette[i])
        for i in range(num_preds):
            yvals = data_box[i]
            xvals = np.random.normal(1 + i, 0.04, size=len(yvals))
            ax2.scatter(xvals, yvals, alpha=0.8, color=palette[i],
                        edgecolor='black', s=50, linewidth=1.5)
        ax2.set_xticks(range(1, num_preds + 1))
        ax2.set_xticklabels([f"P{i}" for i in range(1, num_preds + 1)], fontsize=16)
        ax2.set_xlabel("Predictors", fontsize=20, fontweight='bold')
        ax2.set_ylabel("Score", fontsize=20, fontweight='bold')
        ax2.set_title("Score Distribution Across Simulations", fontsize=24, fontweight='bold')
    else:
        ax2.set_title("No Data for Boxplot", fontsize=20)

    plt.tight_layout()
    save_png(fig, "plot_combined_scores.png")
    plt.show()

def plot_combined_graphs(scores_byD_arbitrary, corrByD_arbitrary,
                         scores_byI_arbitrary, corrByI_arbitrary,
                         scores_byD_high_std, corrByD_high_std,
                         scores_byI_high_std, corrByI_high_std,
                         std_drugs, std_individuals):
    import math, numpy as np, matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    # ──────────────────────────────────────────────────────────────
    def _sd_bar(ax, data, xlabel, title, *, xoff):
        n, vmin, vmax = len(data), float(min(data)), float(max(data))
        norm = Normalize(vmin=vmin, vmax=vmax)

        ax.bar(range(n),
               data,
               color=cm.Greys(0.20 + 0.80 * norm(data)),
               edgecolor='black', linewidth=0.4)

        ax.set_xlabel(xlabel,               fontsize=20, fontweight='bold')
        ax.set_ylabel("Standard Deviation", fontsize=20,
                      fontweight='bold', labelpad=8)
        ax.set_title(title,                 fontsize=22, fontweight='bold', pad=16)
        ax.set_xticks(range(n))
        ax.set_xticklabels(range(1, n + 1), fontsize=15, rotation=90)
        ax.tick_params(axis='y', labelsize=15)
        ax.grid(axis='y', ls=':', alpha=0.35)

        cax = ax.inset_axes([xoff, 0.19, 0.022, 0.62])          # <─ only xoff differs
        sm  = cm.ScalarMappable(norm=norm, cmap=cm.Greys)
        cb  = plt.colorbar(sm, cax=cax, orientation='vertical')
        cb.outline.set_visible(False)
        cb.set_ticks([vmin, vmax])
        cb.set_ticklabels([f"{math.floor(vmin)}", f"{math.ceil(vmax)}"])
        cb.ax.tick_params(labelsize=13, pad=4)

    from matplotlib.ticker import MaxNLocator

    def _plot_trend(ax, xvals, arrA, arrH, title, xlabel):
        arrA, arrH = np.array(arrA, float), np.array(arrH, float)
        arrA = np.nan_to_num(arrA, nan=np.nanmin(arrA))
        arrH = np.nan_to_num(arrH, nan=np.nanmin(arrH))
        ax.plot(xvals, arrA, marker='o', color='#1f77b4', linewidth=2, label='Arbitrary')
        ax.plot(xvals, arrH, marker='o', color='#2ca02c', linewidth=2, label='High STD')
        ax.set_ylim(min(arrA.min(), arrH.min()) - 0.05,
                    max(arrA.max(), arrH.max()) + 0.05)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=12)
        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
        ax.set_ylabel("Correlation", fontsize=16, fontweight='bold')
        ax.grid(ls=':', alpha=0.5)
        ax.legend(frameon=False, fontsize=13)

        # 🔹 Force integer x-ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ──────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(18, 12),
                            gridspec_kw={'hspace': 0.5, 'wspace': 0.28})

    # top row – SD bar-plots
    _sd_bar(axs[0, 0], std_drugs,
            xlabel="Drugs",
            title="Drug-wise Standard Deviation",
            xoff=-0.26)        # unchanged

    _sd_bar(axs[0, 1], std_individuals,
            xlabel="Individuals",
            title="Individual-wise Standard Deviation",
            xoff=-0.23)        # ← moved 0.03 left

    # bottom row – trends
    _plot_trend(axs[1, 0],
                list(range(1, len(corrByD_arbitrary)+1)),
                corrByD_arbitrary, corrByD_high_std,
                "Arbitrary vs High-STD (Drugs)", "Drugs")
    _plot_trend(axs[1, 1],
                list(range(1, len(corrByI_arbitrary)+1)),
                corrByI_arbitrary, corrByI_high_std,
                "Arbitrary vs High-STD (Individuals)", "Individuals")

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    save_png(fig, "plot_combined_graphs.png")
    plt.show()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def plot_static_heatmap(data_dict, title, min_corr):
    """
    Static Spearman-ρ heat-map with three clearly-separated ranges:

        0 – 0.60        → white → light-blue
        0.60 – min_corr → light-blue → dark-blue
        min_corr – 1    → light-red → red

    • Numeric labels appear **only** when ρ ≥ min_corr and are boldfaced.
    • Cells meeting the threshold are outlined in white.
    """
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # ── assemble matrix ------------------------------------------------------
    subset_lbls = list(data_dict.keys())
    x_lbls = sorted({int(lbl.split('i')[1].split('d')[0]) for lbl in subset_lbls})
    y_lbls = sorted({int(lbl.split('d')[1]) for lbl in subset_lbls})
    H = np.zeros((len(y_lbls), len(x_lbls)))
    for lbl, ρ in data_dict.items():
        i = int(lbl.split('i')[1].split('d')[0]) - 1   # 0-based
        d = int(lbl.split('d')[1]) - 1
        H[d, i] = ρ

    # ── build custom three-segment colour map --------------------------------
    eps = 1e-6                               # tiny gap so red segment starts
    cdict = [
        (0.00, '#ffffff'),                   # white
        (0.60, '#cce5ff'),                   # light blue
        (min_corr, '#003c8f'),               # dark  blue
        (min_corr + eps, '#ffd6d6'),         # light red (just above cutoff)
        (1.00, '#b30000')                    # deep  red
    ]
    cmap = LinearSegmentedColormap.from_list('ρ_tri', cdict)

    # ── plot -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))   # match CE heatmap size
    sns.heatmap(H,
                cmap=cmap, vmin=0, vmax=1,
                square=True, cbar=True,
                xticklabels=x_lbls, yticklabels=y_lbls,
                annot=False, linewidths=0)

    # axis tick density (same logic; fonts aligned to CE)
    step_x = max(1, len(x_lbls)//20) or 1
    step_y = max(1, len(y_lbls)//20) or 1
    ax.set_xticks(range(0, len(x_lbls), step_x))
    ax.set_xticklabels(x_lbls[::step_x], fontsize=10, rotation=90)  # ↓ smaller
    ax.set_yticks(range(0, len(y_lbls), step_y))
    ax.set_yticklabels(y_lbls[::step_y], fontsize=10)               # ↓ smaller

    # bold labels + white outline for cells ≥ threshold
    for r in range(len(y_lbls)):
        for c in range(len(x_lbls)):
            ρ = H[r, c]
            if ρ >= min_corr:
                ax.text(c+0.5, r+0.5, f"{ρ:.2f}",
                        ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if ρ >= 0.75 else 'black')
                ax.add_patch(Rectangle((c, r), 1, 1, fill=False,
                                       edgecolor='white', linewidth=2))

    # axis titles & figure title fonts aligned to CE
    ax.set_xlabel('Individuals', fontweight='bold', fontsize=14)
    ax.set_ylabel('Drugs',       fontweight='bold', fontsize=14)
    ax.set_title(title,          fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    save_png(fig, "plot_static_heatmap.png")
    plt.show()


###############################################################################
# COST-EFFECTIVENESS  +  logging helpers
###############################################################################
def plot_cost_effectiveness_heatmap(data_dict, min_corr):
    subset_labels = list(data_dict.keys())
    x_vals = sorted({int(lbl.split('i')[1].split('d')[0]) for lbl in subset_labels})
    y_vals = sorted({int(lbl.split('d')[1]) for lbl in subset_labels})
    cost_mat = np.zeros((len(y_vals), len(x_vals)))
    corr_mat = np.zeros_like(cost_mat)

    for lbl, rho in data_dict.items():
        i_ = int(lbl.split('i')[1].split('d')[0]); d_ = int(lbl.split('d')[1])
        r, c = d_-1, i_-1
        corr_mat[r, c] = rho
        cost_mat[r, c] = (rho/(i_+d_))*100 if rho >= min_corr else 0

    vmax = cost_mat.max()*1.05 if cost_mat.max() else 1
    fig, ax = plt.subplots(figsize=(8, 6))   # keep same size
    im = ax.imshow(cost_mat, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='auto')

    # tick fonts same as before (and now matching static heatmap)
    ax.set_xticks(range(len(x_vals))); ax.set_xticklabels(x_vals, fontsize=10)
    ax.set_yticks(range(len(y_vals))); ax.set_yticklabels(y_vals, fontsize=10)

    # axis titles wording changed: no "#"
    ax.set_xlabel("Individuals", fontsize=14, fontweight='bold')
    ax.set_ylabel("Drugs",       fontsize=14, fontweight='bold')

    ax.set_title("Cost-Effectiveness Heatmap", fontsize=14, fontweight='bold')

    # colorbar without the equation label (legend only)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # (label removed per request)

    for r in range(cost_mat.shape[0]):
        for c in range(cost_mat.shape[1]):
            if corr_mat[r, c] >= min_corr:
                ax.text(c, r, f"{cost_mat[r,c]:.1f}",
                        ha='center', va='center', color="black", fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    save_png(fig, "plot_cost_effectiveness_heatmap.png")
    plt.show()


def log_top_correlation_subsets(data_dict, min_corr,
                                drugs_used_dict, individuals_used_dict, topN=10):
    print("\nTop subsets by Spearman ρ:")
    arr = [(int(k.split('i')[1].split('d')[0]),
            int(k.split('d')[1]), v)
           for k, v in data_dict.items() if v >= min_corr]
    arr.sort(key=lambda x: (-x[2], x[1]+x[0]))
    for idx, (i_, d_, rho) in enumerate(arr[:topN], 1):
        print(f"#{idx:2d}  ρ={rho:.3f}  Drugs={d_:2d}  Indiv={i_:2d}  "
              f"D_list={drugs_used_dict.get(d_, [])}  "
              f"I_list={individuals_used_dict.get(i_, [])}")
    if not arr:
        print("  • none met the threshold •")


def log_top_cost_subsets(data_dict, min_corr,
                         drugs_used_dict, individuals_used_dict, topN=10):
    print("\nTop subsets by cost-effectiveness:")
    arr = []
    for k, rho in data_dict.items():
        if rho < min_corr:
            continue
        i_, d_ = int(k.split('i')[1].split('d')[0]), int(k.split('d')[1])
        cost = rho/(i_+d_)*100
        arr.append((i_, d_, rho, cost))
    arr.sort(key=lambda x: -x[3])
    for idx, (i_, d_, rho, cost) in enumerate(arr[:topN], 1):
        print(f"#{idx:2d}  CE={cost:6.2f}  ρ={rho:.3f}  Drugs={d_:2d}  "
              f"Indiv={i_:2d}  D_list={drugs_used_dict.get(d_, [])}  "
              f"I_list={individuals_used_dict.get(i_, [])}")
    if not arr:
        print("  • none met the threshold •")


def plot_zoomed_in_correlation_heatmap(data_dict, min_corr, topN=10):
    subset_labels = list(data_dict.keys())
    if not subset_labels:
        print("No data to zoom in on."); return
    x_vals = sorted({int(lbl.split('i')[1].split('d')[0]) for lbl in subset_labels})
    y_vals = sorted({int(lbl.split('d')[1]) for lbl in subset_labels})
    corr_mat = np.zeros((len(y_vals), len(x_vals)))
    for lbl, rho in data_dict.items():
        i_, d_ = int(lbl.split('i')[1].split('d')[0]), int(lbl.split('d')[1])
        corr_mat[d_-1, i_-1] = rho
    pos = np.where(corr_mat >= min_corr)
    if not len(pos[0]):
        print("No cell meets the correlation threshold."); return
    center_d, center_i = pos[0][0], pos[1][0]
    half = 10
    r0, r1 = max(0, center_d-half), min(corr_mat.shape[0], center_d+half)
    c0, c1 = max(0, center_i-half), min(corr_mat.shape[1], center_i+half)
    zoom = corr_mat[r0:r1, c0:c1]
    fig = plt.figure(figsize=(8,6), dpi=300)
    sns.heatmap(zoom, cmap='RdYlBu', vmin=0, vmax=1, square=True,
                annot=True, fmt=".2f",
                xticklabels=x_vals[c0:c1], yticklabels=y_vals[r0:r1])
    plt.xlabel("#Individuals"); plt.ylabel("#Drugs")
    plt.title("Zoomed-in ρ Heatmap"); plt.gca().invert_yaxis()
    plt.tight_layout()
    save_png(fig, "plot_zoomed_in_correlation_heatmap.png")
    plt.show()


def plot_zoomed_in_cost_heatmap(data_dict, min_corr, topN=10):
    arr = []
    for k, rho in data_dict.items():
        if rho < min_corr: continue
        i_, d_ = int(k.split('i')[1].split('d')[0]), int(k.split('d')[1])
        cost = rho/(i_+d_)*100
        arr.append((i_, d_, cost))
    if not arr:
        print("No cost-effective subsets."); return
    arr.sort(key=lambda x: -x[2]); arr = arr[:topN]
    xs = sorted({x[0] for x in arr}); ys = sorted({x[1] for x in arr})
    mat = np.zeros((len(ys), len(xs)))
    for i_, d_, ce in arr:
        mat[ys.index(d_), xs.index(i_)] = ce
    fig = plt.figure(figsize=(8,6), dpi=300)
    sns.heatmap(mat, cmap='YlOrRd', vmin=0, vmax=max(mat.max(),1),
                annot=True, fmt=".1f",
                xticklabels=xs, yticklabels=ys, square=True)
    plt.xlabel("#Individuals"); plt.ylabel("#Drugs")
    plt.title("Zoomed Cost-Effectiveness"); plt.gca().invert_yaxis()
    plt.tight_layout()
    save_png(fig, "plot_zoomed_in_cost_heatmap.png")
    plt.show()

###############################################################################
# DASH APP  (Spearman ρ display)
###############################################################################
# ---------------------------------------------------------------------------
def run_dash_app(port=8060):
    subset_labels = list(data_dict_global.keys())
    if not subset_labels:
        print("No data for Dash."); return

    # ---------- build full correlation matrix --------------------------------
    max_i = max(int(lbl.split('i')[1].split('d')[0]) for lbl in subset_labels)
    max_d = max(int(lbl.split('d')[1]) for lbl in subset_labels)
    corr_full = np.zeros((max_d, max_i))
    for lbl, ρ in data_dict_global.items():
        i = int(lbl.split('i')[1].split('d')[0]) - 1
        d = int(lbl.split('d')[1]) - 1
        corr_full[d, i] = ρ

    # ---------- choose 10×10 window centred on the MOST cost-effective cell --
    # (Only change: pick center by max CE = ρ / (i+d) × 100 among cells ≥ threshold)
    cost_full = np.zeros_like(corr_full)
    for r in range(max_d):
        for c in range(max_i):
            ρ = corr_full[r, c]
            if ρ >= min_corr_global:
                cost_full[r, c] = ρ / ((c + 1) + (r + 1)) * 100

    if np.any(cost_full > 0):
        centre_r, centre_c = np.unravel_index(np.argmax(cost_full), cost_full.shape)
    else:
        # fallback to first qualifying correlation cell if no CE-qualified cells
        pos = np.where(corr_full >= min_corr_global)
        if not len(pos[0]):
            print("No subset meets threshold."); return
        centre_r, centre_c = pos[0][0], pos[1][0]

    win = 10
    r0, r1 = max(0, centre_r - win//2), min(max_d, centre_r + win//2 + 1)
    c0, c1 = max(0, centre_c - win//2), min(max_i, centre_c + win//2 + 1)
    corr_sub = corr_full[r0:r1, c0:c1]

    # ---------- cost-effectiveness matrix ------------------------------------
    cost_sub = np.zeros_like(corr_sub)
    for r in range(corr_sub.shape[0]):
        for c in range(corr_sub.shape[1]):
            ρ = corr_sub[r, c]
            if ρ >= min_corr_global:
                cost_sub[r, c] = ρ / ((r0 + r + 1) + (c0 + c + 1)) * 100

    x_lbls = [str(i) for i in range(c0+1, c1+1)]
    y_lbls = [str(d) for d in range(r0+1, r1+1)]

    # ---------- custom correlation colour-scale ------------------------------
    eps = 1e-6
    cs_corr = [
        [0.00, '#ffffff'],
        [0.60, '#cce5ff'],
        [min_corr_global, '#003c8f'],
        [min_corr_global + eps, '#ffd6d6'],
        [1.00, '#b30000']
    ]

    # ---------- figures ------------------------------------------------------
    def make_corr_fig():
        ann = []
        for r in range(corr_sub.shape[0]):
            for c in range(corr_sub.shape[1]):
                ρ = corr_sub[r, c]
                if ρ >= min_corr_global:
                    ann.append(
                        dict(text=f"{ρ:.2f}", x=c, y=r, showarrow=False,
                             font=dict(color='white' if ρ >= 0.75 else 'black',
                                       size=10, family='Arial Black'))
                    )
        return go.Figure(
            data=[go.Heatmap(z=corr_sub, zmin=0, zmax=1,
                             colorscale=cs_corr, hoverinfo='x+y+z')],
            layout=go.Layout(title="Spearman ρ",
                             annotations=ann,
                             xaxis=dict(title="#Individuals",
                                        ticktext=x_lbls,
                                        tickvals=list(range(len(x_lbls))),
                                        tickangle=-45),
                             yaxis=dict(title="#Drugs",
                                        ticktext=y_lbls,
                                        tickvals=list(range(len(y_lbls)))))
        )

    def make_cost_fig():
        ann = []
        for r in range(cost_sub.shape[0]):
            for c in range(cost_sub.shape[1]):
                if corr_sub[r, c] >= min_corr_global:
                    ann.append(dict(text=f"{cost_sub[r,c]:.1f}", x=c, y=r,
                                    showarrow=False,
                                    font=dict(color='white', size=10,
                                              family='Arial Black')))
        return go.Figure(
            data=[go.Heatmap(z=cost_sub, zmin=0,
                             zmax=max(cost_sub.max(), 1),
                             colorscale='YlOrRd', hoverinfo='x+y+z')],
            layout=go.Layout(title="Cost-Effectiveness (ρ / cost ×100)",
                             annotations=ann,
                             xaxis=dict(title="#Individuals",
                                        ticktext=x_lbls,
                                        tickvals=list(range(len(x_lbls))),
                                        tickangle=-45),
                             yaxis=dict(title="#Drugs",
                                        ticktext=y_lbls,
                                        tickvals=list(range(len(y_lbls)))))
        )

    # ---------- Dash app -----------------------------------------------------
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='heat_corr', figure=make_corr_fig(),
                      style={'display': 'inline-block', 'width': '48%'}),
            dcc.Graph(id='heat_cost', figure=make_cost_fig(),
                      style={'display': 'inline-block', 'width': '48%'})
        ]),
        html.Div(id='click-data', style={'whiteSpace': 'pre-line',
                                         'padding': '20px',
                                         'fontSize': '16px',
                                         'fontFamily': 'Courier New'})
    ])

    @app.callback(Output('click-data', 'children'),
                  [Input('heat_corr', 'clickData'),
                   Input('heat_cost', 'clickData')])
    def display_click(corr_click, cost_click):
        ctx = dash.callback_context
        if not ctx.triggered:
            return "Click a cell in either heat-map."
        click = ctx.triggered[0]['value']
        if not click: return "Click a cell in either heat-map."

        r = click['points'][0]['y']
        c = click['points'][0]['x']
        i_val = c0 + c + 1
        d_val = r0 + r + 1
        ρ      = corr_sub[r, c]
        cost   = cost_sub[r, c]

        drug_list = drugs_used_dict_global.get(d_val, [])
        ind_list  = individuals_used_dict_global.get(i_val, [])

        return (f"#Drugs          : {d_val}\n"
                f"#Individuals    : {i_val}\n"
                f"Spearman ρ      : {ρ:.2f}\n"
                f"Cost-Efficiency : {cost:.2f}\n"
                f"Drugs           : {drug_list}\n"
                f"Individuals     : {ind_list}")

    app.run_server(debug=False, use_reloader=False, port=port)
# ---------------------------------------------------------------------------

###############################################################################
# EXECUTION
###############################################################################
if __name__ == "__main__":
    main()

