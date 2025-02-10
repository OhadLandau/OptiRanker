import matplotlib.pyplot as plt
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

###############################################################################
# GLOBALS (will be populated after main runs)
###############################################################################
data_dict_global = {}
scores_byD_high_std_global = {}
scores_byI_high_std_global = {}
min_corr_global = 0.0
drugs_used_dict_global = {}
individuals_used_dict_global = {}

###############################################################################
# MAIN
###############################################################################
def main():
    """
    Main function to run simulations and handle data input.
    """
    num_iterations = int(
        input("Do you want to run multiple simulations? Enter the number of iterations (1 for single simulation): ")
    )

    print("Please enter three dimensions:")
    num_individual = int(input("Enter rows (individuals): "))
    num_predictors = int(input("Enter key (3rd dimension - predictors): "))
    num_drugs = int(input("Enter columns (drugs): "))

    corr = float(input("Enter minimum correlation wanted: "))
    distance = float(input("Enter distance acceptable from correlation: "))

    # Single or multiple
    if num_iterations > 1:
        run_multiple_simulations(num_individual, num_drugs, num_predictors, corr, distance, None, num_iterations)
    else:
        # single
        results, drugs_used_dict, individuals_used_dict = run_project(
            num_individual, num_drugs, num_predictors, corr,
            distance, None, True, True
        )

        if len(results) != 12:
            print(f"Unexpected number of elements in results: {len(results)}")

        # Unpack results
        (
            best_subsetDict,
            scores_byD_arbitrary,
            scores_byI_arbitrary,
            corrByD_arbitrary,
            corrByI_arbitrary,
            best_predictor,
            list_of_stds,
            individual_stds,
            scores_byD_high_std,
            corrByD_high_std,
            scores_byI_high_std,
            corrByI_high_std
        ) = results

        # 1) Combined predictor scores
        plot_combined_scores([best_predictor], best_predictor)

        # 2) Combined graphs
        plot_combined_graphs(
            scores_byD_arbitrary, corrByD_arbitrary,
            scores_byI_arbitrary, corrByI_arbitrary,
            scores_byD_high_std, corrByD_high_std,
            scores_byI_high_std, corrByI_high_std,
            list_of_stds, individual_stds
        )

        # 3) Static heatmap for correlation
        plot_static_heatmap(best_subsetDict, title="Single Simulation Heatmap of Subsets", min_corr=corr)

        # store global references
        global data_dict_global, scores_byD_high_std_global, scores_byI_high_std_global
        global min_corr_global, drugs_used_dict_global, individuals_used_dict_global

        data_dict_global = best_subsetDict
        scores_byD_high_std_global = scores_byD_high_std
        scores_byI_high_std_global = scores_byI_high_std
        min_corr_global = corr
        drugs_used_dict_global = drugs_used_dict
        individuals_used_dict_global = individuals_used_dict

    # After single or multiple runs, produce extra plots/logs and run Dash:
    # 1) Cost-effectiveness heatmap
    plot_cost_effectiveness_heatmap(data_dict_global, min_corr_global)

    # 2) LOG top subsets (correlation, cost)
    log_top_correlation_subsets(
        data_dict_global, min_corr_global,
        drugs_used_dict_global, individuals_used_dict_global, topN=10
    )
    log_top_cost_subsets(
        data_dict_global, min_corr_global,
        drugs_used_dict_global, individuals_used_dict_global, topN=10
    )

    # 3) Zoomed-in static heatmaps
    plot_zoomed_in_correlation_heatmap(data_dict_global, min_corr_global, topN=10)
    plot_zoomed_in_cost_heatmap(data_dict_global, min_corr_global, topN=10)

    # 4) Run the updated Dash app (side-by-side correlation & cost sub-heatmaps) on port 8060
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
        cleaned_rows = []
        for row in data[predictor_key]:
            cleaned_row = [value for value in row if value != 0]
            cleaned_rows.append(cleaned_row)
        cleaned_data[predictor_key] = cleaned_rows
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
            ranking = get_ranking(row)
            ranked_predictor_data.append(ranking)
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
def pearson_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("The lists must have the same length.")
    correlation = np.corrcoef(x, y)[0, 1]
    return correlation

def rank_integers(lst):
    rank_dict = {val: i + 1 for i, val in enumerate(sorted(lst))}
    return [rank_dict[val] for val in lst]

###############################################################################
# SCORING-RELATED FUNCTIONS (CORE LOGIC - UNCHANGED)
###############################################################################
def drug_standardDeviation(d, data, num_individuals, num_predictors):
    list_drug_rank_per_pred = []
    std = 0
    for individual in range(num_individuals):
        for predictor in data:
            list_drug_rank_per_pred.append(data[predictor][individual][d])
        std += np.std(list_drug_rank_per_pred)
        list_drug_rank_per_pred = []
    return std

def RandomDataGenerator(num_individuals, num_drugs, num_predictors, gold_standard=None):
    temp_sim_data = {}
    counter = 0

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

    temp_sim_data["Predictor {0}".format(counter)] = gold_standard
    results_GS = {"Test Results": gold_standard}

    for counter in range(1, num_predictors):
        degraded_predictor = np.copy(temp_sim_data["Predictor {0}".format(counter - 1)])
        num_individuals_to_change = counter
        num_pairs_to_swap = counter

        for _ in range(num_individuals_to_change):
            individual_to_change = random.randint(0, num_individuals - 1)
            for _ in range(num_pairs_to_swap):
                drug1, drug2 = random.sample(range(num_drugs), 2)
                degraded_predictor[individual_to_change][drug1], degraded_predictor[individual_to_change][drug2] = \
                    degraded_predictor[individual_to_change][drug2], degraded_predictor[individual_to_change][drug1]

        temp_sim_data["Predictor {0}".format(counter)] = degraded_predictor

    temp_sim_data["Gold Standard"] = results_GS["Test Results"]
    return temp_sim_data, results_GS

def listTheStds(data, num_individuals, num_predictors, num_drugs):
    list_of_stds = []
    for i in range(num_drugs):
        list_of_stds.append(drug_standardDeviation(i, data, num_individuals, num_predictors))
    return list_of_stds

def drug_selector(d, list_of_stds):
    sorted_indices = sorted(range(len(list_of_stds)), key=lambda i: list_of_stds[i], reverse=True)[:d]
    return sorted_indices

def individual_standardDeviation(data, num_individuals, num_predictors, num_drugs):
    stds = []
    for individual in range(num_individuals):
        values = []
        for predictor in data.keys():
            values.append(data[predictor][individual])
        individual_std = np.sum(np.std(values, axis=0))
        stds.append(individual_std)
    return stds

def individual_selector(i, list_of_stds):
    sorted_indices = sorted(range(len(list_of_stds)), key=lambda idx: list_of_stds[idx], reverse=True)[:i]
    return sorted_indices

###############################################################################
# run_project (CORE LOGIC - UNCHANGED)
###############################################################################
def run_project(num_individual, num_drugs, num_predictors, corr, distance, data,
                select_high_std_drugs=False, select_high_std_individuals=False):
    temp_sim_data = {}
    results_GS = {}
    individual_stds = []

    simulated_Data, results_GS = RandomDataGenerator(num_individual, num_drugs, num_predictors, data)
    print(f"Simulated Data: {simulated_Data}")  # Debug print

    list_of_stds = listTheStds(simulated_Data, num_individual, num_predictors, num_drugs)
    print(f"List of STDs: {list_of_stds}")  # Debug print

    if data is None:
        data = simulated_Data

    # Selection logic
    if select_high_std_drugs:
        drugs_to_filter_by = drug_selector(num_drugs, list_of_stds)
    else:
        drugs_to_filter_by = list(range(num_drugs))

    if select_high_std_individuals:
        individual_stds = individual_standardDeviation(data, num_individual, num_predictors, num_drugs)
        individuals_to_filter_by = individual_selector(num_individual, individual_stds)
    else:
        individuals_to_filter_by = list(range(num_individual))

    print("Selected drugs:", drugs_to_filter_by)  # Debug print
    print("Selected individuals:", individuals_to_filter_by)  # Debug print

    def rank_predictors(data, n_drugs_to_rank, drugs_to_filter, individuals_to_filter):
        n_predictors = len(data.keys())
        n_drugs = len(next(iter(data.values()))[0])
        n_individuals = len(next(iter(data.values())))

        # not used for scoring directly, but the code is from your original script
        sd_drugs = [0] * n_drugs
        for drug in range(n_drugs):
            ranks = [
                data[predictor][ind][drug]
                for predictor in list(data.keys())[:-1]
                for ind in individuals_to_filter
            ]
            ranks = [rank for rank in ranks if isinstance(rank, (int, float))]
            sd_drugs[drug] = np.std(ranks)

        predictor_scores = []
        for predictor in list(data.keys())[:-1]:
            total_diff = 0
            for ind in individuals_to_filter:
                for drug_idx in range(n_drugs_to_rank):
                    if drug_idx in drugs_to_filter:
                        total_diff += (
                            data[predictor][ind][drug_idx]
                            - data[list(data.keys())[-1]][ind][drug_idx]
                        ) ** 2
            predictor_scores.append(total_diff)

        return predictor_scores

    def normalize_and_invert(scores):
        scaler = MinMaxScaler()
        scores = np.array(scores).reshape(-1, 1)
        normalized_scores = scaler.fit_transform(scores)
        inverted_scores = 1 - normalized_scores
        return inverted_scores.flatten().tolist()

    # best predictor
    best_predictor = rank_predictors(data, num_drugs, drugs_to_filter_by, individuals_to_filter_by)
    best_predictor = normalize_and_invert(best_predictor)

    # Arbitrary: scores_byD_arbitrary
    scores_byD_arbitrary = {}
    for i_d in list(range(1, num_drugs + 1)):
        raw_scores = rank_predictors(data, i_d, list(range(i_d)), individuals_to_filter_by)
        scores_byD_arbitrary[f"{i_d}"] = normalize_and_invert(raw_scores)

    best_rank_with_noise = rank_integers(scores_byD_arbitrary[f"{num_drugs}"])
    corrByD_arbitrary = []
    for value in sorted(scores_byD_arbitrary.keys(), key=int):
        correlation = abs(pearson_correlation(best_rank_with_noise,
                                              rank_integers(scores_byD_arbitrary[value])))
        corrByD_arbitrary.append(correlation)
        print(f"Correlation by arbitrary drugs {value}: {correlation}")

    # Arbitrary: scores_byI_arbitrary
    scores_byI_arbitrary = {}
    for i_ct in list(range(1, num_individual + 1)):
        raw_scores = rank_predictors(data, num_drugs, drugs_to_filter_by, list(range(i_ct)))
        scores_byI_arbitrary[f"{i_ct}"] = normalize_and_invert(raw_scores)

    best_rank_with_noise = rank_integers(scores_byI_arbitrary[f"{num_individual}"])
    corrByI_arbitrary = []
    for value in sorted(scores_byI_arbitrary.keys(), key=int):
        correlation = abs(pearson_correlation(best_rank_with_noise,
                                              rank_integers(scores_byI_arbitrary[value])))
        corrByI_arbitrary.append(correlation)
        print(f"Correlation by arbitrary individuals {value}: {correlation}")

    # High STD: scores_byD_high_std
    scores_byD_high_std = {}
    for i_d in list(range(1, num_drugs + 1)):
        selected_drugs = drug_selector(i_d, list_of_stds)
        raw_scores = rank_predictors(data, i_d, selected_drugs, individuals_to_filter_by)
        scores_byD_high_std[f"{i_d}"] = normalize_and_invert(raw_scores)

    best_rank_with_noise = rank_integers(scores_byD_high_std[f"{num_drugs}"])
    corrByD_high_std = []
    for value in sorted(scores_byD_high_std.keys(), key=int):
        correlation = abs(pearson_correlation(best_rank_with_noise,
                                              rank_integers(scores_byD_high_std[value])))
        corrByD_high_std.append(correlation)
        print(f"Correlation by high std drugs {value}: {correlation}")

    # High STD: scores_byI_high_std
    if select_high_std_individuals:
        individual_stds = individual_standardDeviation(data, num_individual, num_predictors, num_drugs)
    scores_byI_high_std = {}
    for i_ct in list(range(1, num_individual + 1)):
        selected_inds = individual_selector(i_ct, individual_stds)
        raw_scores = rank_predictors(data, num_drugs, drugs_to_filter_by, selected_inds)
        scores_byI_high_std[f"{i_ct}"] = normalize_and_invert(raw_scores)

    best_rank_with_noise = rank_integers(scores_byI_high_std[f"{num_individual}"])
    corrByI_high_std = []
    for value in sorted(scores_byI_high_std.keys(), key=int):
        correlation = abs(pearson_correlation(best_rank_with_noise,
                                              rank_integers(scores_byI_high_std[value])))
        corrByI_high_std.append(correlation)
        print(f"Correlation by high std individuals {value}: {correlation}")

    # calculate_scores => your older chunk
    def calculate_scores(i, d, data, num_preds, num_indiv):
        twoD_data = list(chain(*data.values()))
        gs = twoD_data[(num_indiv - 1) * num_preds : num_indiv * num_preds]
        gs_list = [[gs[b][0:d]] for b in range(min(i, len(gs)))]
        list_of_scores = [0] * (num_preds)
        for predictor_index in range(num_preds):
            list_of_scores_temp = []
            for individual_index in range(i):
                list_index = individual_index + predictor_index * num_indiv
                if list_index < len(twoD_data):
                    m = twoD_data[list_index][0:d]
                    if individual_index < len(gs_list):
                        min_len = min(len(m), len(gs_list[individual_index][0]))
                        score = sum(
                            (elem - gs_list[individual_index][0][idx]) ** 2
                            for idx, elem in enumerate(m[:min_len])
                        )
                        list_of_scores_temp.append(score)
            list_of_scores[predictor_index] = sum(list_of_scores_temp)
        return list_of_scores

    def bestSubsetI(i, d, data, num_preds, num_indiv, num_drugs):
        data_subset = data
        # top d drugs by STD, top i individuals by STD
        drug_sel = drug_selector(d, list_of_stds)
        ind_sel = individual_selector(i, individual_stds)

        list_of_scores = calculate_scores(i, d, data_subset, num_preds, num_indiv)
        unique_scores = set()
        for idx, sc in enumerate(list_of_scores):
            while sc in unique_scores:
                sc += 1e-10
            unique_scores.add(sc)
            list_of_scores[idx] = sc
        sorted_scores = sorted(list_of_scores)
        rank_map = {val: idx+1 for idx,val in enumerate(sorted_scores)}
        ranked_scores = [rank_map[sc] for sc in list_of_scores]

        full_scores = calculate_scores(num_indiv, num_drugs, data_subset, num_preds, num_indiv)
        sorted_full_scores = sorted(full_scores)
        rank_full_map = {val: idx+1 for idx,val in enumerate(sorted_full_scores)}
        ranked_full_scores = [rank_full_map[s] for s in full_scores]
        return ranked_scores, ranked_full_scores

    drugs_used_dict = {}
    individuals_used_dict = {}
    best_subsetDict = {}
    num_preds = num_predictors

    # fill best_subsetDict => i=1..N, d=1..N
    if data is None:
        # same logic as your script
        cIn, cDr = 1, 1
        while cIn <= num_individual:
            cDr = 1
            while cDr <= num_drugs:
                sub_ranks, full_ranks = bestSubsetI(cIn, cDr, data, num_preds, num_individual, num_drugs)
                best_subsetDict[f"i{cIn}d{cDr}"] = abs(pearson_correlation(sub_ranks, full_ranks))
                drugs_used_dict[cDr] = drug_selector(cDr, list_of_stds)
                individuals_used_dict[cIn] = individual_selector(cIn, individual_stds)
                cDr += 1
            cIn += 1
    else:
        data_with_noise = data
        cIn, cDr = 1, 1
        while cIn <= num_individual:
            cDr = 1
            while cDr <= num_drugs:
                sub_ranks, full_ranks = bestSubsetI(
                    cIn, cDr, data_with_noise, num_preds, num_individual, num_drugs
                )
                best_subsetDict[f"i{cIn}d{cDr}"] = abs(pearson_correlation(sub_ranks, full_ranks))
                drugs_used_dict[cDr] = drug_selector(cDr, list_of_stds)
                individuals_used_dict[cIn] = individual_selector(cIn, individual_stds)
                cDr += 1
            cIn += 1

    results = (
        best_subsetDict,
        scores_byD_arbitrary,
        scores_byI_arbitrary,
        corrByD_arbitrary,
        corrByI_arbitrary,
        best_predictor,
        list_of_stds,
        individual_stds,
        scores_byD_high_std,
        corrByD_high_std,
        scores_byI_high_std,
        corrByI_high_std
    )
    print(f"Results tuple has {len(results)} elements: {results}")
    return results, drugs_used_dict, individuals_used_dict


###############################################################################
# run_multiple_simulations (CORE LOGIC - UNCHANGED)
###############################################################################
def run_multiple_simulations(num_individual, num_drugs, num_predictors, corr, distance, data, num_iterations):
    all_best_predictor_scores = []
    all_scores_byD_arbitrary = []
    all_corrByD_arbitrary = []
    all_scores_byI_arbitrary = []
    all_corrByI_arbitrary = []
    all_scores_byD_high_std = []
    all_corrByD_high_std = []
    all_scores_byI_high_std = []
    all_corrByI_high_std = []
    all_best_subsetDict = []

    global data_dict_global, scores_byD_high_std_global, scores_byI_high_std_global, min_corr_global
    global drugs_used_dict_global, individuals_used_dict_global

    for _ in range(num_iterations):
        results, d_used, i_used = run_project(
            num_individual, num_drugs, num_predictors, corr,
            distance, data, True, True
        )
        (
            best_subsetDict,
            scores_byD_arbitrary,
            scores_byI_arbitrary,
            corrByD_arbitrary,
            corrByI_arbitrary,
            best_predictor,
            list_of_stds,
            individual_stds,
            scores_byD_high_std,
            corrByD_high_std,
            scores_byI_high_std,
            corrByI_high_std
        ) = results

        # store usage
        drugs_used_dict_global = d_used
        individuals_used_dict_global = i_used

        all_best_predictor_scores.append(best_predictor)
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

    # average dicts
    def average_dicts(dicts):
        from itertools import chain
        all_keys = set(chain.from_iterable(d.keys() for d in dicts))
        avg_dict = {}
        for key in all_keys:
            matching = [d[key] for d in dicts if key in d]
            if matching:
                arr = np.array(matching, dtype=float)
                mean_val = np.mean(arr, axis=0).tolist()
                avg_dict[key] = mean_val
            else:
                avg_dict[key] = []
        return avg_dict

    averaged_scores_byD_arbitrary = average_dicts(all_scores_byD_arbitrary)
    averaged_corrByD_arbitrary = np.mean(all_corrByD_arbitrary, axis=0).tolist()
    averaged_scores_byI_arbitrary = average_dicts(all_scores_byI_arbitrary)
    averaged_corrByI_arbitrary = np.mean(all_corrByI_arbitrary, axis=0).tolist()
    averaged_scores_byD_high_std = average_dicts(all_scores_byD_high_std)
    averaged_corrByD_high_std = np.mean(all_corrByD_high_std, axis=0).tolist()
    averaged_scores_byI_high_std = average_dicts(all_scores_byI_high_std)
    averaged_corrByI_high_std = np.mean(all_corrByI_high_std, axis=0).tolist()
    averaged_best_subsetDict = average_dicts(all_best_subsetDict)

    # plots
    plot_combined_scores(all_best_predictor_scores, averaged_best_predictor_scores)
    plot_combined_graphs(
        averaged_scores_byD_arbitrary, averaged_corrByD_arbitrary,
        averaged_scores_byI_arbitrary, averaged_corrByI_arbitrary,
        averaged_scores_byD_high_std, averaged_corrByD_high_std,
        averaged_scores_byI_high_std, averaged_corrByI_high_std,
        list_of_stds, individual_stds
    )
    plot_static_heatmap(averaged_best_subsetDict, title="Averaged Heatmap of Subsets", min_corr=corr)

    data_dict_global = averaged_best_subsetDict
    scores_byD_high_std_global = averaged_scores_byD_high_std
    scores_byI_high_std_global = averaged_scores_byI_high_std
    min_corr_global = corr


###############################################################################
# PLOTTING (CORE + Additional)
###############################################################################
def plot_combined_scores(all_best_predictor_scores, averaged_best_predictor_scores):
    fig, axs = plt.subplots(2, 1, figsize=(18, 14))

    # top: bar
    ax1 = axs[0]
    avg_scores = averaged_best_predictor_scores
    ax1.bar(range(len(avg_scores)), avg_scores, color=sns.color_palette("Set2", len(avg_scores)))
    ax1.set_xlabel("Predictors", fontsize=20, fontweight='bold')
    ax1.set_ylabel("Average Score", fontsize=20, fontweight='bold')
    ax1.set_title("Predictor Scores", fontsize=24, fontweight='bold')
    ax1.set_xticks(range(len(avg_scores)))
    ax1.set_xticklabels([f"P{i + 1}" for i in range(len(avg_scores))], fontsize=16)

    # bottom: boxplot
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
        ax2.set_title("Box Plot of Predictor Scores Across Simulations", fontsize=24, fontweight='bold')
    else:
        ax2.set_title("No Data for Boxplot", fontsize=20)

    plt.tight_layout()
    plt.show()

def plot_combined_graphs(scores_byD_arbitrary, corrByD_arbitrary,
                         scores_byI_arbitrary, corrByI_arbitrary,
                         scores_byD_high_std, corrByD_high_std,
                         scores_byI_high_std, corrByI_high_std,
                         std_drugs, std_individuals):
    fig, axs = plt.subplots(3, 2, figsize=(18, 21), gridspec_kw={'hspace': 0.7})

    def plot_combined(ax, scores_by, corrBy, x_axis, title):
        data = []
        if isinstance(scores_by, dict):
            for num, scores in sorted(scores_by.items(), key=lambda x: int(x[0])):
                total_score = sum(scores)
                if total_score == 0:
                    normalized_scores = [0]*len(scores)
                else:
                    normalized_scores = [v/total_score for v in scores]
                for i_idx, sc in enumerate(normalized_scores):
                    data.append([int(num), f"P{i_idx+1}", sc])
        else:
            # fallback if it was a list
            for i_idx, scores_item in enumerate(scores_by):
                total_score = sum(scores_item)
                if total_score == 0:
                    normalized_scores = [0]*len(scores_item)
                else:
                    normalized_scores = [v/total_score for v in scores_item]
                for j_idx, sc in enumerate(normalized_scores):
                    data.append([i_idx+1, f"P{j_idx+1}", sc])

        df = pd.DataFrame(data, columns=[x_axis, "Predictor", "Score"])
        df['Predictor'] = df['Predictor'].apply(lambda x: int(x[1:]))

        df.sort_values(by=['Predictor'], inplace=True)
        pivot_df = df.pivot_table(index=x_axis, columns='Predictor', values='Score', fill_value=0)

        bottom = np.zeros(len(pivot_df))
        palette = sns.color_palette("Set2", len(pivot_df.columns))
        for i_col, col in enumerate(pivot_df.columns):
            ax.bar(pivot_df.index, pivot_df[col],
                   bottom=bottom,
                   label=f'P{col}',
                   color=palette[i_col])
            bottom += pivot_df[col]

        max_ticks = 20
        step_ = max(1, len(pivot_df.index)//max_ticks) if len(pivot_df.index)>0 else 1
        ax.set_xticks(pivot_df.index[::step_])
        ax.set_xticklabels(pivot_df.index[::step_], fontsize=12, rotation=90)

        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_xlabel(x_axis, fontsize=16, fontweight='bold')
        ax.set_ylabel("Normalized Score", fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)

    # row0 col0 => arbitrary drugs
    plot_combined(axs[0,0], scores_byD_arbitrary, corrByD_arbitrary, "Drugs",
                  "Scores and Correlation (Arbitrary Drugs)")
    # row1 col0 => high std drugs
    plot_combined(axs[1,0], scores_byD_high_std, corrByD_high_std, "Drugs",
                  "Scores and Correlation (High Std Drugs)")

    # row0 col1 => arbitrary individuals
    plot_combined(axs[0,1], scores_byI_arbitrary, corrByI_arbitrary, "Individuals",
                  "Scores and Correlation (Arbitrary Individuals)")
    # row1 col1 => high std individuals
    plot_combined(axs[1,1], scores_byI_high_std, corrByI_high_std, "Individuals",
                  "Scores and Correlation (High Std Individuals)")

    def plot_trendline(ax, xvals, arrA, arrH, title, xlabel):
        arrA = np.array(arrA, dtype=float)
        arrH = np.array(arrH, dtype=float)

        min_arbitrary = np.nanmin(arrA) if np.isnan(arrA).any() else arrA.min()
        min_highstd = np.nanmin(arrH) if np.isnan(arrH).any() else arrH.min()

        arrA = np.where(np.isnan(arrA), min_arbitrary, arrA)
        arrH = np.where(np.isnan(arrH), min_highstd, arrH)

        ax.plot(xvals, arrA, marker='o', color='b', linewidth=2, label='Arbitrary')
        ax.plot(xvals, arrH, marker='o', color='g', linewidth=2, label='High Std')

        y_min = min(arrA.min(), arrH.min())
        y_max = max(arrA.max(), arrH.max())
        ax.set_ylim(y_min-0.1, y_max+0.1)
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
        ax.set_ylabel("Correlation", fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)
        ax.legend()

    # row2 => correlation lines
    x_drugs = list(range(1, len(corrByD_arbitrary)+1))
    plot_trendline(axs[2,0], x_drugs, corrByD_arbitrary, corrByD_high_std,
                   "Arbitrary vs High Std Selection of Drugs", "Number of Drugs")

    x_inds = list(range(1, len(corrByI_arbitrary)+1))
    plot_trendline(axs[2,1], x_inds, corrByI_arbitrary, corrByI_high_std,
                   "Arbitrary vs High Std Selection of Individuals", "Number of Individuals")

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(14, len(handles)), fontsize=14)
    plt.tight_layout(rect=[0,0.05,1,0.95])
    plt.show()

    # separate figure => STD barplots
    fig2, axs2 = plt.subplots(2,1, figsize=(18,14))

    def plot_std_barplot(ax, data, title, xlabel, color):
        ax.bar(range(len(data)), data, color=color)
        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
        ax.set_ylabel("Standard Deviation", fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)

    plot_std_barplot(axs2[0], std_drugs, "Standard Deviations of Drugs", "Drugs", color='skyblue')
    plot_std_barplot(axs2[1], std_individuals, "Standard Deviations of Individuals", "Individuals", color='lightcoral')

    plt.tight_layout()
    plt.show()

def plot_static_heatmap(data_dict, title, min_corr):
    """
    Create a static heatmap highlighting subsets >= min_corr
    """
    subset_labels = list(data_dict.keys())
    subset_values = list(data_dict.values())

    x_labels = sorted(list(set(int(lbl.split('i')[1].split('d')[0]) for lbl in subset_labels)))
    y_labels = sorted(list(set(int(lbl.split('d')[1]) for lbl in subset_labels)))

    heatmap_data = np.zeros((len(y_labels), len(x_labels)))
    for lbl, val in zip(subset_labels, subset_values):
        i_ = int(lbl.split('i')[1].split('d')[0])
        d_ = int(lbl.split('d')[1])
        heatmap_data[d_-1, i_-1] = val

    fig_width = min(20, len(x_labels)/2)
    fig_height = min(20, len(y_labels)/2)
    plt.figure(figsize=(fig_width, fig_height))

    ax = sns.heatmap(
        heatmap_data, cmap='Blues', cbar=False, square=True,
        xticklabels=x_labels, yticklabels=y_labels
    )

    max_xticks = 20
    max_yticks = 20
    x_tick_step = max(1, len(x_labels)//max_xticks) if len(x_labels)>0 else 1
    y_tick_step = max(1, len(y_labels)//max_yticks) if len(y_labels)>0 else 1
    ax.set_xticks(range(0, len(x_labels), x_tick_step))
    ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), x_tick_step)], fontsize=8, rotation=90)
    ax.set_yticks(range(0, len(y_labels), y_tick_step))
    ax.set_yticklabels([y_labels[i] for i in range(0, len(y_labels), y_tick_step)], fontsize=8)

    for row_ in range(len(y_labels)):
        for col_ in range(len(x_labels)):
            val = heatmap_data[row_, col_]
            if val >= min_corr:
                ax.add_patch(
                    plt.Rectangle((col_, row_),1,1, fill=False, edgecolor='white', linewidth=2)
                )
                ax.text(
                    col_+0.5, row_+0.5, f"{val:.2f}",
                    ha='center', va='center', color='white',
                    fontsize=7, fontweight='bold'
                )

    plt.xlabel('Individuals', fontsize=18, fontweight='bold')
    plt.ylabel('Drugs', fontsize=18, fontweight='bold')
    plt.title(title, fontsize=20, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

###############################################################################
# COST-EFFECTIVENESS & Additional Logging
###############################################################################
def plot_cost_effectiveness_heatmap(data_dict, min_corr):
    """
    cost_effective(%) = (corr/(X + Y))*100 if corr >= min_corr else 0
    highlight top5, annotate if corr>=0.7
    """
    subset_labels = list(data_dict.keys())
    x_values = sorted(list(set(int(lbl.split('i')[1].split('d')[0]) for lbl in subset_labels)))
    y_values = sorted(list(set(int(lbl.split('d')[1]) for lbl in subset_labels)))
    cost_matrix = np.zeros((len(y_values), len(x_values)))
    corr_matrix = np.zeros((len(y_values), len(x_values)))

    for lbl, corr_val in data_dict.items():
        i_val = int(lbl.split('i')[1].split('d')[0])
        d_val = int(lbl.split('d')[1])
        xx = x_values.index(i_val)
        yy = y_values.index(d_val)
        if corr_val >= min_corr:
            cost_matrix[yy, xx] = (corr_val/(i_val + d_val))*100
        else:
            cost_matrix[yy, xx] = 0
        corr_matrix[yy, xx] = corr_val

    vmax = cost_matrix.max()*1.05 if cost_matrix.max()>0 else 1
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cost_matrix, cmap="YlOrRd", vmin=0, vmax=vmax, aspect='auto')

    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_xticklabels(x_values, fontsize=10)
    ax.set_yticklabels(y_values, fontsize=10)
    ax.set_xlabel("Number of Individuals", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Drugs", fontsize=12, fontweight='bold')
    ax.set_title("Cost-Effective Heatmap", fontsize=14, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cost-Effective", fontsize=12, fontweight='bold')

    # highlight top5
    coords_list = [(row, col) for row in range(cost_matrix.shape[0]) for col in range(cost_matrix.shape[1])]
    sorted_ = sorted(coords_list, key=lambda idx: cost_matrix[idx[0], idx[1]], reverse=True)
    top5 = sorted_[:5]
    for (r_, c_) in top5:
        rect = plt.Rectangle((c_-0.5, r_-0.5),1,1, fill=False, edgecolor="black", linewidth=3)
        ax.add_patch(rect)

    # annotate if corr>=0.7
    for rr in range(cost_matrix.shape[0]):
        for cc in range(cost_matrix.shape[1]):
            if corr_matrix[rr, cc]>=0.7:
                ax.text(cc, rr, f"{cost_matrix[rr, cc]:.1f}",
                        ha='center', va='center', color="black",
                        fontsize=8, fontweight='bold')

    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

def log_top_correlation_subsets(data_dict, min_corr, drugs_used_dict, individuals_used_dict, topN=10):
    print("Logging top correlation subsets...")
    arr = []
    for k,v in data_dict.items():
        i_ = int(k.split('i')[1].split('d')[0])
        d_ = int(k.split('d')[1])
        if v >= min_corr:
            arr.append((i_, d_, v))
    arr.sort(key=lambda x: (x[1], x[0]))
    top = arr[:topN]
    if not top:
        print(f"No subsets found with correlation >= {min_corr}.")
        return
    for idx, (ii_, dd_, corr_) in enumerate(top):
        print("-------------------------------")
        print(f"Subset {idx+1}:")
        print(f"  Number of Drugs: {dd_}")
        print(f"  Number of Individuals: {ii_}")
        print(f"  Drugs: {drugs_used_dict.get(dd_, [])}")
        print(f"  Individuals: {individuals_used_dict.get(ii_, [])}")
        print(f"  Correlation: {corr_:.2f}")
    print("-------------------------------")

def log_top_cost_subsets(data_dict, min_corr, drugs_used_dict, individuals_used_dict, topN=10):
    print("Logging top cost-effectiveness subsets...")
    arr = []
    for k,v in data_dict.items():
        i_ = int(k.split('i')[1].split('d')[0])
        d_ = int(k.split('d')[1])
        if v >= min_corr:
            cost = (v/(i_+d_))*100
            arr.append((i_, d_, v, cost))
    arr.sort(key=lambda x: x[3], reverse=True)
    top = arr[:topN]
    if not top:
        print(f"No subsets found with correlation >= {min_corr} for cost-effectiveness.")
        return
    for idx,(ii_, dd_, cval, cost_) in enumerate(top):
        print("-------------------------------")
        print(f"Subset {idx+1}:")
        print(f"  Number of Drugs: {dd_}")
        print(f"  Number of Individuals: {ii_}")
        print(f"  Drugs: {drugs_used_dict.get(dd_, [])}")
        print(f"  Individuals: {individuals_used_dict.get(ii_, [])}")
        print(f"  Correlation: {cval:.2f}")
        print(f"  Cost-Effectiveness: {cost_:.2f}")
    print("-------------------------------")

def plot_zoomed_in_correlation_heatmap(data_dict, min_corr, topN=10):
    print("Plotting zoomed-in correlation heatmap for top subsets by correlation...")
    arr = []
    for k,v in data_dict.items():
        i_ = int(k.split('i')[1].split('d')[0])
        d_ = int(k.split('d')[1])
        if v >= min_corr:
            arr.append((i_, d_, v))
    arr.sort(key=lambda x: (x[1], x[0]))
    top = arr[:topN]
    if not top:
        print(f"No subsets found with correlation >= {min_corr}.")
        return

    min_i = min(x[0] for x in top)
    max_i = max(x[0] for x in top)
    min_d = min(x[1] for x in top)
    max_d = max(x[1] for x in top)
    h = max_d - min_d +1
    w = max_i - min_i +1
    sub_mat = np.zeros((h,w))
    for (iv,dv,corr_) in top:
        row = dv-min_d
        col = iv-min_i
        sub_mat[row, col] = corr_

    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    sns.heatmap(
        sub_mat, cmap='Blues', vmin=0, vmax=1, cbar=True,
        square=True, annot=True, fmt=".2f",
        xticklabels=range(min_i, max_i+1),
        yticklabels=range(min_d, max_d+1),
        ax=ax
    )
    ax.set_xlabel("Number of Individuals", fontsize=12)
    ax.set_ylabel("Number of Drugs", fontsize=12)
    ax.set_title("Zoomed Correlation (Top Subsets)", fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_zoomed_in_cost_heatmap(data_dict, min_corr, topN=10):
    print("Plotting zoomed-in cost-effectiveness heatmap for top subsets by cost...")
    arr = []
    for k,v in data_dict.items():
        i_ = int(k.split('i')[1].split('d')[0])
        d_ = int(k.split('d')[1])
        if v >= min_corr:
            cost = (v/(i_+d_))*100
            arr.append((i_, d_, v, cost))
    arr.sort(key=lambda x: x[3], reverse=True)
    top = arr[:topN]
    if not top:
        print(f"No subsets found for cost-effectiveness >= {min_corr}.")
        return

    min_i = min(x[0] for x in top)
    max_i = max(x[0] for x in top)
    min_d = min(x[1] for x in top)
    max_d = max(x[1] for x in top)
    h = max_d - min_d +1
    w = max_i - min_i +1
    sub_mat = np.zeros((h,w))

    for (iv, dv, corr_, cost_) in top:
        row = dv-min_d
        col = iv-min_i
        sub_mat[row, col] = cost_

    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    vmax = sub_mat.max() if sub_mat.max()>0 else 1
    sns.heatmap(
        sub_mat, cmap='YlOrRd', vmin=0, vmax=vmax, cbar=True,
        square=True, annot=True, fmt=".2f",
        xticklabels=range(min_i, max_i+1),
        yticklabels=range(min_d, max_d+1),
        ax=ax
    )
    ax.set_xlabel("Number of Individuals", fontsize=12)
    ax.set_ylabel("Number of Drugs", fontsize=12)
    ax.set_title("Zoomed Cost-Effectiveness (Top Subsets)", fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

###############################################################################
# DASH APP: Side-by-side Correlation & Cost sub-heatmaps
###############################################################################
def run_dash_app(port=8060):
    """
    Build a 10x10 window around the best correlation cell in data_dict_global:
      Left: correlation sub-heatmap
      Right: cost sub-heatmap
    Then run on port=8060.
    """
    subset_labels = list(data_dict_global.keys())
    if not subset_labels:
        print("No data available for the Dash app. (data_dict_global is empty.)")
        return

    # find max i, d
    max_i = 0
    max_d = 0
    for lbl in subset_labels:
        i_val = int(lbl.split('i')[1].split('d')[0])
        d_val = int(lbl.split('d')[1])
        max_i = max(max_i, i_val)
        max_d = max(max_d, d_val)

    # Build correlation matrix
    corr_matrix_full = np.zeros((max_d, max_i))
    for lbl, val in data_dict_global.items():
        i_val = int(lbl.split('i')[1].split('d')[0])
        d_val = int(lbl.split('d')[1])
        corr_matrix_full[d_val - 1, i_val - 1] = val

    # best subset by correlation
    best_key = max(data_dict_global, key=data_dict_global.get)
    best_i = int(best_key.split('i')[1].split('d')[0])
    best_d = int(best_key.split('d')[1])

    window_size = 10

    # figure out submatrix bounds for i
    start_i = best_i - window_size//2
    end_i = start_i + window_size
    if start_i < 1:
        start_i = 1
        end_i = start_i + window_size
    if end_i > max_i:
        end_i = max_i
        start_i = max(end_i - window_size + 1, 1)

    # figure out submatrix bounds for d
    start_d = best_d - window_size//2
    end_d = start_d + window_size
    if start_d < 1:
        start_d = 1
        end_d = start_d + window_size
    if end_d > max_d:
        end_d = max_d
        start_d = max(end_d - window_size + 1, 1)

    # slice submat
    corr_sub = corr_matrix_full[start_d-1:end_d-1, start_i-1:end_i-1]
    cost_sub = np.zeros_like(corr_sub)
    for row in range(corr_sub.shape[0]):
        for col in range(corr_sub.shape[1]):
            d_idx = start_d + row
            i_idx = start_i + col
            cval = corr_matrix_full[d_idx - 1, i_idx - 1]
            if cval >= min_corr_global:
                cost_val = (cval/(i_idx + d_idx))*100.0
            else:
                cost_val = 0.0
            cost_sub[row, col] = cost_val

    # build x,y labels for dash
    x_labels = [str(i) for i in range(start_i, end_i)]
    y_labels = [str(d) for d in range(start_d, end_d)]
    sub_height = corr_sub.shape[0]
    sub_width = corr_sub.shape[1]

    # correlation figure
    annotations_corr = []
    zmin_corr = corr_sub.min()
    zmax_corr = corr_sub.max()
    for r_ in range(sub_height):
        for c_ in range(sub_width):
            val = corr_sub[r_, c_]
            text_color = "white" if val >= min_corr_global else "black"
            annotations_corr.append(dict(
                text=f"{val:.2f}",
                x=c_,
                y=r_,
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
        margin=dict(l=100, r=50, t=70, b=80)
    )
    fig_corr = go.Figure(data=[heatmap_corr], layout=layout_corr)

    # cost figure
    annotations_cost = []
    zmin_cost = cost_sub.min()
    zmax_cost = cost_sub.max() if cost_sub.max()>0 else 1
    for r_ in range(sub_height):
        for c_ in range(sub_width):
            val = cost_sub[r_, c_]
            corr_val = corr_sub[r_, c_]
            text_color = "white" if corr_val >= min_corr_global else "black"
            annotations_cost.append(dict(
                text=f"{val:.2f}",
                x=c_,
                y=r_,
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
        zmax=zmax_cost,
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
        margin=dict(l=100, r=50, t=70, b=80)
    )
    fig_cost = go.Figure(data=[heatmap_cost], layout=layout_cost)

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='heatmap_corr', figure=fig_corr, style={'display': 'inline-block', 'width': '45%'}),
            dcc.Graph(id='heatmap_cost', figure=fig_cost, style={'display': 'inline-block', 'width': '45%'})
        ]),
        html.Div(
            id='click-data-corr',
            style={'whiteSpace': 'pre-line','padding':'20px','font-size':'16px',
                   'width':'45%','display':'inline-block','verticalAlign':'top'}
        ),
        html.Div(
            id='click-data-cost',
            style={'whiteSpace':'pre-line','padding':'20px','font-size':'16px',
                   'width':'45%','display':'inline-block','verticalAlign':'top'}
        )
    ])

    @app.callback(
        Output('click-data-corr','children'),
        [Input('heatmap_corr','clickData')]
    )
    def display_click_data_corr(clickData):
        if not clickData:
            return "Click on a cell in the Correlation Heatmap."
        pt = clickData['points'][0]
        x_sub = int(pt['x'])
        y_sub = int(pt['y'])

        i_val = start_i + x_sub
        d_val = start_d + y_sub
        corr_val = corr_matrix_full[d_val - 1, i_val - 1]

        used_drugs = drugs_used_dict_global.get(d_val, [])
        used_inds = individuals_used_dict_global.get(i_val, [])

        info = (
            f"Subset => #Drugs = {d_val}, #Individuals = {i_val}\n"
            f"Correlation: {corr_val:.2f}\n"
            f"Drugs: {used_drugs}\n"
            f"Individuals: {used_inds}\n"
        )
        return info

    @app.callback(
        Output('click-data-cost','children'),
        [Input('heatmap_cost','clickData')]
    )
    def display_click_data_cost(clickData):
        if not clickData:
            return "Click on a cell in the Cost-Effectiveness Heatmap."
        pt = clickData['points'][0]
        x_sub = int(pt['x'])
        y_sub = int(pt['y'])

        i_val = start_i + x_sub
        d_val = start_d + y_sub
        corr_val = corr_matrix_full[d_val - 1, i_val - 1]
        cost_val = 0.0
        if corr_val >= min_corr_global:
            cost_val = (corr_val/(i_val + d_val))*100.0

        used_drugs = drugs_used_dict_global.get(d_val, [])
        used_inds = individuals_used_dict_global.get(i_val, [])

        info = (
            f"Subset => #Drugs = {d_val}, #Individuals = {i_val}\n"
            f"Correlation: {corr_val:.2f}\n"
            f"Cost-Effectiveness: {cost_val:.2f}\n"
            f"Drugs: {used_drugs}\n"
            f"Individuals: {used_inds}\n"
        )
        return info

    app.run_server(debug=True, use_reloader=False, port=port)

###############################################################################
# EXECUTION
###############################################################################
if __name__ == "__main__":
    main()
