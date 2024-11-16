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

def main():
    """
    Main function to run simulations and handle data input.
    """
    num_iterations = int(input("Do you want to run multiple simulations? Enter the number of iterations (1 for single simulation): "))

    print("Please enter three dimensions:")
    num_individual = int(input("Enter rows (individuals): "))
    num_predictors = int(input("Enter key (3rd dimension - predictors): "))
    num_drugs = int(input("Enter columns (drugs): "))

    corr = float(input("Enter minimum correlation wanted: "))
    distance = float(input("Enter distance acceptable from correlation: "))

    if num_iterations > 1:
        run_multiple_simulations(num_individual, num_drugs, num_predictors, corr, distance, None, num_iterations)
    else:
        results, drugs_used_dict, individuals_used_dict = run_project(num_individual, num_drugs, num_predictors, corr, distance, None, True, True)
        if len(results) != 12:
            print(f"Unexpected number of elements in results: {len(results)}")
        plot_combined_scores([results[5]], results[5])  # Pass a list with a single element for consistency
        plot_combined_graphs(results[1], results[3], results[2], results[4],
                             results[9], results[10], results[11], results[8],
                             results[6], results[7])
        plot_static_heatmap(results[0], title="Single Simulation Heatmap of Subsets", min_corr=corr)
        # Remove the call to plot_interactive_heatmap here
        # The Dash app will be run in the __main__ block

        # Store necessary variables for the Dash app
        global data_dict_global, scores_byD_high_std_global, scores_byI_high_std_global, min_corr_global, drugs_used_dict_global, individuals_used_dict_global
        data_dict_global = results[0]
        scores_byD_high_std_global = results[9]
        scores_byI_high_std_global = results[10]
        min_corr_global = corr
        drugs_used_dict_global = drugs_used_dict
        individuals_used_dict_global = individuals_used_dict

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

def pearson_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("The lists must have the same length.")
    correlation = np.corrcoef(x, y)[0, 1]
    return correlation

def rank_integers(lst):
    rank_dict = {val: i + 1 for i, val in enumerate(sorted(lst))}
    return [rank_dict[val] for val in lst]

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

def run_project(num_individual, num_drugs, num_predictors, corr, distance, data, select_high_std_drugs=False, select_high_std_individuals=False):
    temp_sim_data = {}
    results_GS = {}
    individual_stds = []

    simulated_Data, results_GS = RandomDataGenerator(num_individual, num_drugs, num_predictors, data)
    print(f"Simulated Data: {simulated_Data}")  # Debug print

    list_of_stds = listTheStds(simulated_Data, num_individual, num_predictors, num_drugs)
    print(f"List of STDs: {list_of_stds}")  # Debug print

    if data is None:
        data = simulated_Data

    # Selection logic for drugs and individuals
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

        sd_drugs = [0] * n_drugs
        for drug in range(n_drugs):
            ranks = [data[predictor][ind][drug] for predictor in list(data.keys())[:-1] for ind in individuals_to_filter]
            ranks = [rank for rank in ranks if isinstance(rank, (int, float))]
            sd_drugs[drug] = np.std(ranks)

        predictor_scores = []
        for predictor in list(data.keys())[:-1]:
            total_diff = 0
            for ind in individuals_to_filter:
                for drug in range(n_drugs_to_rank):
                    if drug in drugs_to_filter:
                        total_diff += (data[predictor][ind][drug] - data[list(data.keys())[-1]][ind][drug]) ** 2
            predictor_scores.append(total_diff)

        return predictor_scores

    def normalize_and_invert(scores):
        scaler = MinMaxScaler()
        scores = np.array(scores).reshape(-1, 1)
        normalized_scores = scaler.fit_transform(scores)
        inverted_scores = 1 - normalized_scores
        return inverted_scores.flatten().tolist()

    best_predictor = rank_predictors(data, num_drugs, drugs_to_filter_by, individuals_to_filter_by)
    best_predictor = normalize_and_invert(best_predictor)

    scores_byD_arbitrary = {}
    for i in list(range(1, num_drugs + 1)):
        scores_byD_arbitrary[f"{i}"] = rank_predictors(data, i, list(range(i)), individuals_to_filter_by)
        scores_byD_arbitrary[f"{i}"] = normalize_and_invert(scores_byD_arbitrary[f"{i}"])
    best_rank_with_noise = rank_integers(scores_byD_arbitrary[f"{num_drugs}"])
    corrByD_arbitrary = []
    for value in sorted(scores_byD_arbitrary.keys(), key=int):  # Ensure keys are sorted numerically
        correlation = abs(pearson_correlation(best_rank_with_noise, rank_integers(scores_byD_arbitrary[value])))
        corrByD_arbitrary.append(correlation)
        print(f"Correlation by arbitrary drugs {value}: {correlation}")  # Debug print

    scores_byI_arbitrary = {}
    for i in list(range(1, num_individual + 1)):
        scores_byI_arbitrary[f"{i}"] = rank_predictors(data, num_drugs, drugs_to_filter_by, list(range(i)))
        scores_byI_arbitrary[f"{i}"] = normalize_and_invert(scores_byI_arbitrary[f"{i}"])
    best_rank_with_noise = rank_integers(scores_byI_arbitrary[f"{num_individual}"])
    corrByI_arbitrary = []
    for value in sorted(scores_byI_arbitrary.keys(), key=int):  # Ensure keys are sorted numerically
        correlation = abs(pearson_correlation(best_rank_with_noise, rank_integers(scores_byI_arbitrary[value])))
        corrByI_arbitrary.append(correlation)
        print(f"Correlation by arbitrary individuals {value}: {correlation}")  # Debug print

    # Calculate individual standard deviations
    if select_high_std_individuals:
        individual_stds = individual_standardDeviation(data, num_individual, num_predictors, num_drugs)

    scores_byD_high_std = {}
    for i in list(range(1, num_drugs + 1)):
        selected_drugs = drug_selector(i, list_of_stds)  # Use drug_selector here
        scores_byD_high_std[f"{i}"] = rank_predictors(data, i, selected_drugs, individuals_to_filter_by)
        scores_byD_high_std[f"{i}"] = normalize_and_invert(scores_byD_high_std[f"{i}"])
    best_rank_with_noise = rank_integers(scores_byD_high_std[f"{num_drugs}"])
    corrByD_high_std = []
    for value in sorted(scores_byD_high_std.keys(), key=int):  # Ensure keys are sorted numerically
        correlation = abs(pearson_correlation(best_rank_with_noise, rank_integers(scores_byD_high_std[value])))
        corrByD_high_std.append(correlation)
        print(f"Correlation by high std drugs {value}: {correlation}")  # Debug print

    scores_byI_high_std = {}
    for i in list(range(1, num_individual + 1)):
        selected_inds = individual_selector(i, individual_stds)  # Select individuals by highest std
        scores_byI_high_std[f"{i}"] = rank_predictors(data, num_drugs, drugs_to_filter_by, selected_inds)
        scores_byI_high_std[f"{i}"] = normalize_and_invert(scores_byI_high_std[f"{i}"])
    best_rank_with_noise = rank_integers(scores_byI_high_std[f"{num_individual}"])
    corrByI_high_std = []
    for value in sorted(scores_byI_high_std.keys(), key=int):  # Ensure keys are sorted numerically
        correlation = abs(pearson_correlation(best_rank_with_noise, rank_integers(scores_byI_high_std[value])))
        corrByI_high_std.append(correlation)
        print(f"Correlation by high std individuals {value}: {correlation}")  # Debug print

    def calculate_scores(i, d, data, num_preds, num_indiv):
        twoD_data = list(chain(*data.values()))
        gs = twoD_data[(num_indiv - 1) * num_preds: num_indiv * num_preds]
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
                            [(elem - gs_list[individual_index][0][idx]) ** 2 for idx, elem in enumerate(m[:min_len])])
                        list_of_scores_temp.append(score)
            list_of_scores[predictor_index] = sum(list_of_scores_temp)
        return list_of_scores

    def bestSubsetI(i, d, data, num_preds, num_indiv, num_drugs):
        data_subset = data
        drugs_to_filter = drug_selector(d, list_of_stds)  # Select drugs by highest std
        individuals_to_filter = individual_selector(i, individual_stds)  # Select individuals by highest std
        list_of_scores = calculate_scores(i, d, data_subset, num_preds, num_indiv)
        unique_scores = set()
        for idx, score in enumerate(list_of_scores):
            while score in unique_scores:
                score += 1e-10
            unique_scores.add(score)
            list_of_scores[idx] = score
        sorted_scores = sorted(list_of_scores)
        rank = {val: idx + 1 for idx, val in enumerate(sorted_scores)}
        ranked_scores = [rank[score] for score in list_of_scores]

        full_scores = calculate_scores(num_indiv, num_drugs, data, num_preds, num_indiv)
        sorted_full_scores = sorted(full_scores)
        rank_full = {val: idx + 1 for idx, val in enumerate(sorted_full_scores)}
        ranked_full_scores = [rank_full[score] for score in full_scores]
        return ranked_scores, ranked_full_scores

    drugs_used_dict = {}
    individuals_used_dict = {}

    if data is None:
        cIn, cDr = 1, 1
        best_subsetDict = {}
        list_for_tempSubset = []
        while cIn <= num_individual:
            cDr = 1
            while cDr <= num_drugs:
                list_for_tempSubset, list_of_bestRanks = bestSubsetI(cIn, cDr, data, num_predictors, num_individual,
                                                                     num_drugs)
                best_subsetDict["i{}d{}".format(cIn, cDr)] = abs(
                    pearson_correlation(list_for_tempSubset, list_of_bestRanks))
                drugs_used_dict[cDr] = drug_selector(cDr, list_of_stds)
                individuals_used_dict[cIn] = individual_selector(cIn, individual_stds)
                cDr += 1
            cIn += 1
    else:
        data_with_noise = data
        cIn, cDr = 1, 1
        best_subsetDict = {}
        list_for_tempSubset = []
        while cIn <= num_individual:
            cDr = 1
            while cDr <= num_drugs:
                list_for_tempSubset, list_of_bestRanks = bestSubsetI(cIn, cDr, data_with_noise, num_predictors,
                                                                     num_individual, num_drugs)
                best_subsetDict["i{}d{}".format(cIn, cDr)] = abs(
                    pearson_correlation(list_for_tempSubset, list_of_bestRanks))
                drugs_used_dict[cDr] = drug_selector(cDr, list_of_stds)
                individuals_used_dict[cIn] = individual_selector(cIn, individual_stds)
                cDr += 1
            cIn += 1

    results = (best_subsetDict, scores_byD_arbitrary, scores_byI_arbitrary, corrByD_arbitrary, corrByI_arbitrary,
               best_predictor, list_of_stds, individual_stds, scores_byD_high_std, corrByD_high_std, scores_byI_high_std,
               corrByI_high_std)

    print(f"Results tuple has {len(results)} elements: {results}")  # Debug print
    return results, drugs_used_dict, individuals_used_dict

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

    for _ in range(num_iterations):
        results, drugs_used_dict, individuals_used_dict = run_project(num_individual, num_drugs, num_predictors, corr, distance, data, True, True)
        best_subsetDict, scores_byD_arbitrary, scores_byI_arbitrary, corrByD_arbitrary, corrByI_arbitrary, best_predictor, list_of_stds, individual_stds, scores_byD_high_std, corrByD_high_std, scores_byI_high_std, corrByI_high_std = results
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

    def average_dicts(dicts):
        all_keys = set(chain.from_iterable(d.keys() for d in dicts))
        avg_dict = {}
        for key in all_keys:
            avg_dict[key] = np.mean([d[key] for d in dicts if key in d], axis=0).tolist()
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

    plot_combined_scores(all_best_predictor_scores, averaged_best_predictor_scores)
    plot_combined_graphs(averaged_scores_byD_arbitrary, averaged_corrByD_arbitrary, averaged_scores_byI_arbitrary, averaged_corrByI_arbitrary,
                         averaged_scores_byD_high_std, averaged_corrByD_high_std, averaged_scores_byI_high_std, averaged_corrByI_high_std, list_of_stds,
                         individual_stds)

    plot_static_heatmap(averaged_best_subsetDict, title="Averaged Heatmap of Subsets", min_corr=corr)
    # Remove the call to plot_interactive_heatmap here

    # The Dash app will be run in the __main__ block

    # Store necessary variables for the Dash app
    global data_dict_global, scores_byD_high_std_global, scores_byI_high_std_global, min_corr_global, drugs_used_dict_global, individuals_used_dict_global
    data_dict_global = averaged_best_subsetDict
    scores_byD_high_std_global = averaged_scores_byD_high_std
    scores_byI_high_std_global = averaged_scores_byI_high_std
    min_corr_global = corr
    drugs_used_dict_global = drugs_used_dict
    individuals_used_dict_global = individuals_used_dict

def plot_combined_scores(all_best_predictor_scores, averaged_best_predictor_scores):
    fig, axs = plt.subplots(2, 1, figsize=(18, 14))

    ax1 = axs[0]
    avg_scores = averaged_best_predictor_scores
    ax1.bar(range(len(avg_scores)), avg_scores, color=sns.color_palette("Set2", len(avg_scores)))
    ax1.set_xlabel("Predictors", fontsize=20, fontweight='bold')
    ax1.set_ylabel("Average Score", fontsize=20, fontweight='bold')
    ax1.set_title("Predictor Scores", fontsize=24, fontweight='bold')
    ax1.set_xticks(range(len(avg_scores)))
    ax1.set_xticklabels([f"P{i + 1}" for i in range(len(avg_scores))], fontsize=16)

    ax2 = axs[1]
    data = []
    num_predictors = len(all_best_predictor_scores[0])
    for i in range(num_predictors):
        predictor_scores = [simulation[i] for simulation in all_best_predictor_scores]
        data.append(predictor_scores)

    box = ax2.boxplot(data, patch_artist=True, showmeans=True, meanline=True)

    palette = sns.color_palette("Set2", len(avg_scores))
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor('none')
        patch.set_edgecolor(palette[i])
        patch.set_linewidth(2)
        for element in ['whiskers', 'caps', 'medians', 'means']:
            plt.setp(box[element][2 * i:2 * (i + 1)], color=palette[i], linewidth=2)
        plt.setp(box['fliers'][i], markerfacecolor=palette[i], markeredgecolor=palette[i])

    for i in range(num_predictors):
        y = data[i]
        x = np.random.normal(1 + i, 0.04, size=len(y))
        ax2.scatter(x, y, alpha=0.8, color=palette[i], edgecolor='black', s=50, linewidth=1.5)

    ax2.set_xticks(range(1, num_predictors + 1))
    ax2.set_xticklabels([f"P{i}" for i in range(1, num_predictors + 1)], fontsize=16)
    ax2.set_xlabel("Predictors", fontsize=20, fontweight='bold')
    ax2.set_ylabel("Score", fontsize=20, fontweight='bold')
    ax2.set_title("Box Plot of Predictor Scores Across Simulations", fontsize=24, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_combined_graphs(scores_byD_arbitrary, corrByD_arbitrary, scores_byI_arbitrary, corrByI_arbitrary,
                         scores_byD_high_std, corrByD_high_std, scores_byI_high_std, corrByI_high_std, std_drugs,
                         std_individuals):
    fig, axs = plt.subplots(3, 2, figsize=(18, 21), gridspec_kw={'hspace': 0.7})

    def plot_combined(ax, scores_by, corrBy, x_axis, title):
        data = []
        if isinstance(scores_by, dict):
            for num, scores in sorted(scores_by.items(), key=lambda x: int(x[0])):  # Sort by numeric order
                total_score = sum(scores)
                if total_score == 0:
                    normalized_scores = [0] * len(scores)
                else:
                    normalized_scores = [score / total_score for score in scores]
                for i, score in enumerate(normalized_scores):
                    data.append([int(num), f"P{i + 1}", score])  # Ensure num is an integer
        else:
            for i, scores in enumerate(scores_by):
                total_score = sum(scores)
                if total_score == 0:
                    normalized_scores = [0] * len(scores)
                else:
                    normalized_scores = [score / total_score for score in scores]
                for j, score in enumerate(normalized_scores):
                    data.append([i + 1, f"P{j + 1}", score])

        scores_df = pd.DataFrame(data, columns=[x_axis, "Predictor", "Score"])
        scores_df['Predictor'] = scores_df['Predictor'].apply(lambda x: int(x[1:]))
        scores_df = scores_df.sort_values(by=['Predictor'])
        scores_pivot = scores_df.pivot_table(index=x_axis, columns='Predictor', values='Score', fill_value=0)
        corr_df = pd.DataFrame({x_axis: list(range(1, len(corrBy) + 1)), 'Correlation': corrBy})

        bottom = np.zeros(len(scores_pivot))
        for col in scores_pivot.columns:
            ax.bar(scores_pivot.index, scores_pivot[col], bottom=bottom, label=f'P{col}', color=sns.color_palette("Set2", len(scores_pivot.columns))[col - 1])
            bottom += scores_pivot[col]

        # Limit number of ticks and labels
        max_ticks = 20
        tick_step = max(1, len(scores_pivot.index) // max_ticks)
        ax.set_xticks(scores_pivot.index[::tick_step])
        ax.set_xticklabels(scores_pivot.index[::tick_step], fontsize=12, rotation=90)

        ax.set_xlabel(x_axis, fontsize=16, fontweight='bold')
        ax.set_ylabel("Normalized Score", fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)

    plot_combined(axs[0, 0], scores_byD_arbitrary, corrByD_arbitrary, "Drugs", "Scores and Correlation (Arbitrary Drugs)")
    plot_combined(axs[1, 0], scores_byD_high_std, corrByD_high_std, "Drugs", "Scores and Correlation (High Std Drugs)")

    plot_combined(axs[0, 1], scores_byI_arbitrary, corrByI_arbitrary, "Individuals", "Scores and Correlation (Arbitrary Individuals)")
    plot_combined(axs[1, 1], scores_byI_high_std, corrByI_high_std, "Individuals", "Scores and Correlation (High Std Individuals)")

    def plot_trendline(ax, x, y_arbitrary, y_high_std, title, xlabel):
        y_arbitrary = np.array(y_arbitrary)
        y_high_std = np.array(y_high_std)

        # Replace NaN values with the minimum of existing values for visual continuity, if applicable
        min_arbitrary = np.nanmin(y_arbitrary) if np.isnan(y_arbitrary).any() else min(y_arbitrary)
        min_high_std = np.nanmin(y_high_std) if np.isnan(y_high_std).any() else min(y_high_std)

        y_arbitrary = np.where(np.isnan(y_arbitrary), min_arbitrary, y_arbitrary)
        y_high_std = np.where(np.isnan(y_high_std), min_high_std, y_high_std)

        ax.plot(x, y_arbitrary, marker='o', color='b', linewidth=2, label='Arbitrary')
        ax.plot(x, y_high_std, marker='o', color='g', linewidth=2, label='High Std')

        # Calculate the min and max across both trendlines
        y_min = min(np.min(y_arbitrary), np.min(y_high_std))
        y_max = max(np.max(y_arbitrary), np.max(y_high_std))

        ax.set_ylim(y_min - 0.1, y_max + 0.1)  # Adding some padding

        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
        ax.set_ylabel("Correlation", fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend()
        ax.grid(True)

    x_drugs = list(range(1, len(corrByD_arbitrary) + 1))
    y_drugs_arbitrary = corrByD_arbitrary
    y_drugs_high_std = corrByD_high_std

    x_individuals = list(range(1, len(corrByI_arbitrary) + 1))
    y_individuals_arbitrary = corrByI_arbitrary
    y_individuals_high_std = corrByI_high_std

    plot_trendline(axs[2, 0], x_drugs, y_drugs_arbitrary, y_drugs_high_std, "Arbitrary vs High Std Selection of Drugs", "Number of Drugs")
    plot_trendline(axs[2, 1], x_individuals, y_individuals_arbitrary, y_individuals_high_std, "Arbitrary vs High Std Selection of Individuals", "Number of Individuals")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(14, len(handles)), fontsize=14)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(18, 14))

    def plot_std_barplot(ax, data, title, xlabel, color):
        ax.bar(range(len(data)), data, color=color)
        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
        ax.set_ylabel("Standard Deviation", fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)

    plot_std_barplot(axs[0], std_drugs, "Standard Deviations of Drugs", "Drugs", color='skyblue')
    plot_std_barplot(axs[1], std_individuals, "Standard Deviations of Individuals", "Individuals", color='lightcoral')

    plt.tight_layout()
    plt.show()

def plot_static_heatmap(data_dict, title, min_corr):
    subset_labels = list(data_dict.keys())
    subset_values = list(data_dict.values())

    x_labels = sorted(list(set(int(label.split('i')[1].split('d')[0]) for label in subset_labels)))
    y_labels = sorted(list(set(int(label.split('d')[1]) for label in subset_labels)))

    heatmap_data = np.zeros((len(y_labels), len(x_labels)))

    for label, value in zip(subset_labels, subset_values):
        i = int(label.split('i')[1].split('d')[0])
        d = int(label.split('d')[1])
        heatmap_data[d - 1, i - 1] = value

    # Adjust figure size dynamically
    fig_width = min(20, len(x_labels) / 2)
    fig_height = min(20, len(y_labels) / 2)
    plt.figure(figsize=(fig_width, fig_height))

    ax = sns.heatmap(heatmap_data, cmap='Blues', cbar=False, square=True, xticklabels=x_labels, yticklabels=y_labels)

    # Limit number of ticks and labels
    max_xticks = 20  # Maximum number of x-axis ticks
    max_yticks = 20  # Maximum number of y-axis ticks

    x_tick_step = max(1, len(x_labels) // max_xticks)
    y_tick_step = max(1, len(y_labels) // max_yticks)

    ax.set_xticks(range(0, len(x_labels), x_tick_step))
    ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), x_tick_step)], fontsize=8, rotation=90)
    ax.set_yticks(range(0, len(y_labels), y_tick_step))
    ax.set_yticklabels([y_labels[i] for i in range(0, len(y_labels), y_tick_step)], fontsize=8)

    # Add rectangle patches for values above the correlation threshold
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            value = heatmap_data[i, j]
            if value >= min_corr:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='white', linewidth=2))
                ax.text(j + 0.5, i + 0.5, f"{value:.2f}", ha='center', va='center', color='white', fontsize=7, fontweight='bold')

    # Set axis labels
    plt.xlabel('Individuals', fontsize=18, fontweight='bold')
    plt.ylabel('Drugs', fontsize=18, fontweight='bold')

    # Invert the y-axis to match previous heatmap orientation
    plt.gca().invert_yaxis()

    # Ensure the layout is tight and compact
    plt.tight_layout()

    # Show the plot
    plt.show()

# No need to define plot_interactive_heatmap as a function anymore
# We will move the Dash app code into the __main__ block

if __name__ == "__main__":
    main()

    # Now, define and run the Dash app
    app = dash.Dash(__name__)

    # Prepare data for the interactive heatmap
    data_dict = data_dict_global
    scores_byD_high_std = scores_byD_high_std_global
    scores_byI_high_std = scores_byI_high_std_global
    min_corr = min_corr_global
    drugs_used_dict = drugs_used_dict_global
    individuals_used_dict = individuals_used_dict_global

    # Prepare data for heatmap
    subset_labels = list(data_dict.keys())
    subset_values = list(data_dict.values())

    x_labels = sorted(list(set(int(label.split('i')[1].split('d')[0]) for label in subset_labels)))
    y_labels = sorted(list(set(int(label.split('d')[1]) for label in subset_labels)))

    heatmap_data = np.zeros((len(y_labels), len(x_labels)))

    for label, value in zip(subset_labels, subset_values):
        i = int(label.split('i')[1].split('d')[0])
        d = int(label.split('d')[1])
        heatmap_data[d - 1, i - 1] = value

    annotations = []
    for d in range(len(y_labels)):
        for i in range(len(x_labels)):
            value = heatmap_data[d, i]
            if value >= min_corr:
                annotations.append(dict(
                    text=f"<b>{value:.2f}</b>",
                    x=i,
                    y=d,
                    xref='x1',
                    yref='y1',
                    showarrow=False,
                    font=dict(
                        color="white" if value < 0.5 else "black",
                        size=12,
                        family='Arial'
                    )
                ))

    app.layout = html.Div([
        dcc.Graph(
            id='heatmap',
            figure=go.Figure(
                data=go.Heatmap(
                    z=heatmap_data,
                    x=[str(i) for i in x_labels],
                    y=[str(d) for d in y_labels],
                    colorscale='Blues',
                    showscale=True,
                    text=[[f"{d}d, {i}i" for i in x_labels] for d in y_labels],
                    hoverinfo="text",
                    zmin=0,
                    zmax=1
                ),
                layout=go.Layout(
                    annotations=annotations,
                    xaxis=dict(
                        title='Number of Individuals',
                        tickmode='array',
                        tickvals=list(range(len(x_labels))),
                        ticktext=[str(i) for i in x_labels],
                        tickangle=-45,  # Tilt the x-axis labels
                        automargin=True,
                        tickfont=dict(size=15)
                    ),
                    yaxis=dict(
                        title='Number of Drugs',
                        tickmode='array',
                        tickvals=list(range(len(y_labels))),
                        ticktext=[str(d) for d in y_labels],
                        automargin=True,
                        tickfont=dict(size=15)
                    ),
                    margin=dict(l=100, r=50, t=50, b=100),  # Increase margins for better fit
                )
            )
        ),
        html.Div(id='click-data', style={'whiteSpace': 'pre-line', 'padding': '20px', 'font-size': '16px'})
    ])


    @app.callback(
        Output('click-data', 'children'),
        [Input('heatmap', 'clickData')]
    )
    def display_click_data(clickData):
        if clickData is None:
            return "Click on a heatmap cell to see the details."

        point = clickData['points'][0]
        i = int(point['x'])
        d = int(point['y'])

        # Fetch the first `d` drugs based on highest standard deviations
        drugs = drugs_used_dict.get(d + 1, [])

        # Fetch the first `i` individuals (arbitrarily from 0 to `i`)
        individuals = list(range(i + 1))

        subset_info = f"Drugs: {', '.join(str(x) for x in drugs)}\nIndividuals: {', '.join(str(x) for x in individuals)}\nCorrelation: {heatmap_data[d, i]:.2f}"
        return subset_info


    app.run_server(debug=True, use_reloader=False)