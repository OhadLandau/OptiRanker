import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random
from scipy.stats import pearsonr
from sklearn.impute import KNNImputer
import PySimpleGUI as sg
import seaborn as sns

# Ensure plots are displayed with high resolution
plt.rcParams['figure.dpi'] = 300

# KNN imputation
def knn_impute_data(df, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df)
    return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)

# Preprocess data
def preprocess_data(predictor_files, results_file=None, is_ranked=False, no_results=False):
    data_dict = {}
    predictor_dfs = []
    predictor_names = []

    for idx, file_path in enumerate(predictor_files):
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

    if not no_results and results_file is not None:
        df_results = pd.read_csv(results_file, index_col=0)
        df_results.columns = df_results.columns.str.replace('.', '-', regex=False)
        if df_results.index[0] == "DRUG_NAME":
            df_results = df_results.drop(df_results.index[0])

        common_drugs &= set(df_results.index)
        common_individuals &= set(df_results.columns)

    common_drugs = sorted(list(common_drugs))
    common_individuals = sorted(list(common_individuals))

    aligned_predictor_dfs = [df.loc[common_drugs, common_individuals] for df in predictor_dfs]

    if not is_ranked:
        ranked_predictor_dfs = []
        for df in aligned_predictor_dfs:
            df = df.apply(pd.to_numeric, errors='coerce')
            df_imputed = knn_impute_data(df)
            df_ranked = df_imputed.rank(axis=0, method='min', na_option='keep').astype(int)
            ranked_predictor_dfs.append(df_ranked)
    else:
        ranked_predictor_dfs = [df.loc[common_drugs, common_individuals].rank(axis=0, method='min', na_option='keep').astype(int) for df in aligned_predictor_dfs]

    if not no_results and results_file is not None:
        df_results = df_results.loc[common_drugs, common_individuals]
        if not is_ranked:
            df_results = df_results.apply(pd.to_numeric, errors='coerce')
            df_results_imputed = knn_impute_data(df_results)
            df_results_ranked = df_results_imputed.rank(axis=0, method='min', na_option='keep').astype(int)
        else:
            df_results_ranked = df_results.rank(axis=0, method='min', na_option='keep').astype(int)
    else:
        df_results_ranked = None

    data_dict = {}
    for idx, df in enumerate(ranked_predictor_dfs):
        data_dict[f"Predictor{idx}"] = df.values.tolist()

    return data_dict, common_drugs, common_individuals, df_results_ranked, ranked_predictor_dfs, predictor_names

# Random Data Generator for Simulation
def RandomDataGenerator(num_individuals, num_drugs, num_predictors):
    temp_sim_data = {}
    counter = 0

    unique_ints = np.arange(1, num_drugs + 1)
    gold_standard = np.empty((num_drugs, num_individuals), dtype=int)
    for i in range(num_individuals):
        np.random.shuffle(unique_ints)
        gold_standard[:, i] = unique_ints

    temp_sim_data["Predictor{0}".format(counter)] = gold_standard.copy()

    for counter in range(1, num_predictors):
        degraded_predictor = gold_standard.copy()
        num_swaps = int(0.1 * num_drugs * num_individuals)  # Introduce 10% noise
        for _ in range(num_swaps):
            ind = random.randint(0, num_individuals - 1)
            drug1, drug2 = random.sample(range(num_drugs), 2)
            degraded_predictor[drug1, ind], degraded_predictor[drug2, ind] = degraded_predictor[drug2, ind], degraded_predictor[drug1, ind]
        temp_sim_data["Predictor{0}".format(counter)] = degraded_predictor

    temp_sim_data["GoldStandard"] = gold_standard

    return temp_sim_data, gold_standard

# Compute standard deviations for drugs
def compute_drug_stds(predictor_dfs):
    # Collect rankings for each predictor
    rankings = []
    for df in predictor_dfs:
        rankings.append(df.values)
    # Stack along a new axis
    rankings = np.stack(rankings, axis=-1)  # Shape: (num_drugs, num_individuals, num_predictors)
    # Compute standard deviation for each drug across all individuals and predictors
    drug_stds = np.std(rankings, axis=(1, 2))  # Axis 1: individuals, Axis 2: predictors
    # Create a Series with drug indices
    drug_stds = pd.Series(drug_stds, index=predictor_dfs[0].index)
    return drug_stds

# Compute standard deviations for individuals
def compute_individual_stds(predictor_dfs):
    # Collect rankings for each predictor
    rankings = []
    for df in predictor_dfs:
        rankings.append(df.values)
    # Stack along a new axis
    rankings = np.stack(rankings, axis=-1)  # Shape: (num_drugs, num_individuals, num_predictors)
    # Compute standard deviation for each individual across all drugs and predictors
    individual_stds = np.std(rankings, axis=(0, 2))  # Axis 0: drugs, Axis 2: predictors
    # Create a Series with individual indices
    individual_stds = pd.Series(individual_stds, index=predictor_dfs[0].columns)
    return individual_stds

def pearson_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("The lists must have the same length.")
    correlation = np.corrcoef(x, y)[0, 1]
    return correlation

# Calculate L2 distances (squared differences)
def calculate_scores_full(df_results_ranked, predictor_dfs):
    scores = []
    for df in predictor_dfs:
        differences = df_results_ranked - df
        score = np.nansum(differences ** 2)
        scores.append(score)
    return np.array(scores)

# Simulation function with adjusted correlation logic
def run_project_simulation(num_individuals, num_drugs, num_predictors, min_correlation, N, common_drugs, common_individuals, predictor_names, num_iterations):
    all_subsets = []
    total_iterations = 0

    for iteration in range(num_iterations):
        total_iterations += 1
        temp_sim_data, gold_standard = RandomDataGenerator(num_individuals, num_drugs, num_predictors)

        # Convert simulated data to DataFrames
        predictor_dfs_sim = []
        predictor_names_sim = []
        for key in temp_sim_data.keys():
            if key != 'GoldStandard':
                df = pd.DataFrame(temp_sim_data[key], index=common_drugs[:num_drugs],
                                  columns=common_individuals[:num_individuals])
                predictor_dfs_sim.append(df)
                predictor_names_sim.append(key)
        gold_standard_df = pd.DataFrame(temp_sim_data['GoldStandard'],
                                        index=common_drugs[:num_drugs],
                                        columns=common_individuals[:num_individuals])

        # Compute standard deviations of drugs
        drug_stds = compute_drug_stds(predictor_dfs_sim)
        sorted_drugs = drug_stds.sort_values(ascending=False).index.tolist()

        # Compute standard deviations of individuals
        individual_stds = compute_individual_stds(predictor_dfs_sim)
        sorted_individuals = individual_stds.sort_values(ascending=False).index.tolist()

        subsets_found = 0

        max_num_drugs = num_drugs
        max_num_individuals = num_individuals

        correlations_matrix = np.zeros((max_num_drugs, max_num_individuals))

        for num_drug in range(1, max_num_drugs + 1):
            top_drugs = sorted_drugs[:num_drug]  # Highest std drugs
            for num_indiv in range(1, max_num_individuals + 1):
                top_individuals = sorted_individuals[:num_indiv]  # Highest std individuals

                gold_standard_subset = gold_standard_df.loc[top_drugs, top_individuals]
                predictor_dfs_subset = [df.loc[top_drugs, top_individuals] for df in predictor_dfs_sim]

                # For each predictor, compute the average ranking per drug over the individuals
                gold_standard_mean_rank = gold_standard_subset.mean(axis=1)
                gold_standard_rank = gold_standard_mean_rank.rank(method='min')

                # For each predictor, compute the mean ranking per drug and then rank them
                predictor_mean_ranks = []
                for predictor_df in predictor_dfs_subset:
                    predictor_mean_rank = predictor_df.mean(axis=1)
                    predictor_rank = predictor_mean_rank.rank(method='min')
                    predictor_mean_ranks.append(predictor_rank)

                # For each predictor, compute Pearson correlation with the gold standard
                correlations = []
                for predictor_rank in predictor_mean_ranks:
                    correlation = pearson_correlation(predictor_rank.values, gold_standard_rank.values)
                    if np.isnan(correlation):
                        correlation = 0
                    correlations.append(correlation)

                # Average the correlations over the predictors
                avg_correlation = np.mean(correlations)
                correlations_matrix[num_drug - 1, num_indiv - 1] = avg_correlation

                subset = {
                    'Drugs': top_drugs,
                    'Individuals': top_individuals,
                    'Correlation': avg_correlation,
                    'NumDrugs': num_drug,
                    'NumIndividuals': num_indiv
                }
                all_subsets.append(subset)
                subsets_found += 1

        # Only need to run once since we're simulating
        break

    # Aggregate subsets and select top N subsets based on average correlation
    subsets_df = pd.DataFrame(all_subsets)
    grouped_subsets = subsets_df.groupby(['NumDrugs', 'NumIndividuals']).agg({
        'Correlation': 'mean',
        'Drugs': 'first',
        'Individuals': 'first'
    }).reset_index()

    # Find the subsets that meet the minimum correlation
    valid_subsets = grouped_subsets[grouped_subsets['Correlation'] >= min_correlation]
    if valid_subsets.empty:
        print(f"No subsets found with average correlation >= {min_correlation}")
        num_drugs_first = None
        num_individuals_first = None
    else:
        # Sort by NumDrugs and NumIndividuals to find the first N subsets that meet the criteria
        valid_subsets = valid_subsets.sort_values(by=['NumDrugs', 'NumIndividuals'])
        top_subsets = valid_subsets.head(N)
        num_drugs_first = top_subsets.iloc[0]['NumDrugs']
        num_individuals_first = top_subsets.iloc[0]['NumIndividuals']

        # Print the first N subsets
        for idx, subset in top_subsets.iterrows():
            print(f"Subset {idx + 1}:")
            print(f"Number of Drugs: {subset['NumDrugs']}")
            print(f"Number of Individuals: {subset['NumIndividuals']}")
            print(f"Drugs: {subset['Drugs']}")
            print(f"Individuals: {subset['Individuals']}")
            print(f"Average Correlation: {subset['Correlation']}")
            print("-------------------------------")

    # Return the correlations matrix and subsets DataFrame for plotting
    return correlations_matrix, num_drugs_first, num_individuals_first, grouped_subsets

# Optimize ranking correlation
def optimize_ranking_correlation(df_results_ranked, predictor_dfs, min_correlation, N):
    # Compute the full data scores
    full_data_scores = calculate_scores_full(df_results_ranked, predictor_dfs)
    # Convert full data scores to ranks (lowest score is rank 1)
    full_data_ranking = np.argsort(full_data_scores) + 1  # Adding 1 to make ranks start from 1

    num_drugs_total = len(df_results_ranked.index)
    num_individuals_total = len(df_results_ranked.columns)

    # Compute standard deviations of drugs
    drug_stds = compute_drug_stds(predictor_dfs)
    sorted_drugs = drug_stds.sort_values(ascending=False).index.tolist()

    # Use individuals arbitrarily (since we don't have stds for individuals)
    sorted_individuals = df_results_ranked.columns.tolist()

    correlations = np.zeros((num_drugs_total, num_individuals_total))
    drugs_used_dict, individuals_used_dict = {}, {}
    all_subsets = []

    for num_drugs in range(1, num_drugs_total + 1):
        top_drugs = sorted_drugs[:num_drugs]  # Highest std drugs
        drugs_used_dict[num_drugs] = top_drugs
        for num_individuals in range(1, num_individuals_total + 1):
            top_individuals = sorted_individuals[:num_individuals]  # Arbitrary selection
            individuals_used_dict[num_individuals] = top_individuals

            df_results_subset = df_results_ranked.loc[top_drugs, top_individuals]
            predictor_dfs_subset = [df.loc[top_drugs, top_individuals] for df in predictor_dfs]

            # Calculate the scores for this subset
            subset_scores = calculate_scores_full(df_results_subset, predictor_dfs_subset)
            # Convert subset scores to ranks (lowest score is rank 1)
            subset_ranking = np.argsort(subset_scores) + 1  # Adding 1 to make ranks start from 1

            # Compute Pearson correlation between the two rankings
            correlation = pearson_correlation(subset_ranking, full_data_ranking)
            if np.isnan(correlation):
                correlation = 0
            else:
                correlation = max(0, correlation)
            correlations[num_drugs - 1, num_individuals - 1] = correlation

            # Store the subset information
            subset = {
                'NumDrugs': num_drugs,
                'NumIndividuals': num_individuals,
                'Correlation': correlation,
                'Drugs': top_drugs,
                'Individuals': top_individuals
            }
            all_subsets.append(subset)

    # Create a DataFrame of subsets
    subsets_df = pd.DataFrame(all_subsets)

    # Find the subsets that meet the minimum correlation
    valid_subsets = subsets_df[subsets_df['Correlation'] >= min_correlation]

    if valid_subsets.empty:
        print(f"No subsets found with correlation >= {min_correlation}")
        num_drugs_first = None
        num_individuals_first = None
    else:
        # Sort by NumDrugs and NumIndividuals to find the first N subsets that meet the criteria
        valid_subsets = valid_subsets.sort_values(by=['NumDrugs', 'NumIndividuals'])
        top_subsets = valid_subsets.head(N)
        num_drugs_first = top_subsets.iloc[0]['NumDrugs']
        num_individuals_first = top_subsets.iloc[0]['NumIndividuals']

        # Print the first N subsets
        for idx, subset in top_subsets.iterrows():
            print(f"Subset {idx + 1}:")
            print(f"Number of Drugs: {subset['NumDrugs']}")
            print(f"Number of Individuals: {subset['NumIndividuals']}")
            print(f"Drugs: {subset['Drugs']}")
            print(f"Individuals: {subset['Individuals']}")
            print(f"Correlation: {subset['Correlation']}")
            print("-------------------------------")

    return correlations, num_drugs_first, num_individuals_first, subsets_df
def plot_static_heatmap(correlations_matrix, min_corr, num_drugs_first, num_individuals_first, subsets_df, N):
    # First heatmap: Abstract image without numbers or axis labels
    plt.figure(figsize=(10, 8))
    correlations_matrix_reversed = correlations_matrix[::-1, :]  # Reverse the order of drugs
    sns.heatmap(correlations_matrix_reversed, cmap='Blues', cbar=False, square=True, xticklabels=False, yticklabels=False)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Second heatmap: Zoomed-in view around the first subset that meets the minimum correlation
    if correlations_matrix is not None and num_drugs_first is not None and num_individuals_first is not None:
        # Get indices of the first subset
        idx_drug = num_drugs_first - 1
        idx_indiv = num_individuals_first - 1

        window_size = 20
        start_drug_idx = max(idx_drug - window_size // 2, 0)
        end_drug_idx = min(start_drug_idx + window_size, correlations_matrix.shape[0])
        start_indiv_idx = max(idx_indiv - window_size // 2, 0)
        end_indiv_idx = min(start_indiv_idx + window_size, correlations_matrix.shape[1])

        zoomed_data = correlations_matrix[start_drug_idx:end_drug_idx, start_indiv_idx:end_indiv_idx]
        zoomed_data_reversed = zoomed_data[::-1, :]  # Reverse the order of drugs for visualization
        x_labels = list(range(start_indiv_idx + 1, end_indiv_idx + 1))
        y_labels = list(range(start_drug_idx + 1, end_drug_idx + 1))
        y_labels_reversed = y_labels[::-1]

        # Create a mask to only annotate the first N subsets
        annot_data = np.full_like(zoomed_data_reversed, '', dtype=object)
        top_subsets = subsets_df.head(N)
        for idx, subset in top_subsets.iterrows():
            num_drugs = subset['NumDrugs']
            num_individuals = subset['NumIndividuals']
            if start_drug_idx <= num_drugs - 1 < end_drug_idx and start_indiv_idx <= num_individuals - 1 < end_indiv_idx:
                # Calculate positions in the zoomed_data_reversed
                drug_pos = end_drug_idx - num_drugs  # Since we reversed the y-axis
                indiv_pos = num_individuals - start_indiv_idx - 1
                annot_data[drug_pos - start_drug_idx, indiv_pos] = f"{subset['Correlation']:.2f}"

        plt.figure(figsize=(14, 12))
        ax = sns.heatmap(zoomed_data_reversed, cmap='Blues', cbar=False, square=True,
                         xticklabels=x_labels, yticklabels=y_labels_reversed, annot=annot_data, fmt="", annot_kws={"size": 10})
        ax.set_xlabel('Number of Individuals', fontsize=12)
        ax.set_ylabel('Number of Drugs', fontsize=12)
        plt.title(f"Heatmap Zoomed In Around Subset {num_drugs_first}d{num_individuals_first}i", fontsize=14)
        plt.tight_layout()
        plt.show()

# Plotting functions (same as before)
def plot_bar_chart_with_significance(scores, predictor_labels, p_values):
    buffer = 0.1
    max_score = max(scores)
    min_score = min(scores)
    range_score = max_score - min_score

    if range_score != 0:
        scaled_scores = [1 - (buffer + (score - min_score) / range_score * (1 - 2 * buffer)) for score in scores]
    else:
        scaled_scores = [0.5 for _ in scores]

    plt.figure(figsize=(15, 11))
    colors = sns.color_palette("Set2", len(scaled_scores))
    bars = plt.bar(predictor_labels, scaled_scores, color=colors)
    # Set font size for x-axis labels
    plt.xticks(fontsize=20)
    plt.ylim([0, 1])
    plt.xlabel('Predictors', fontsize=26)
    plt.ylabel('Score (Scaled)', fontsize=26)
    plt.title('Predictor Scores', fontsize=30, fontweight='bold')
    plt.tight_layout()
    plt.show()

def apply_pca_and_plot(df_results_ranked, predictor_dfs, predictor_names, common_individuals):
    # Combine the results and all predictors into a single DataFrame
    combined_data = pd.concat([df_results_ranked] + predictor_dfs, axis=1)

    # Standardize the data
    combined_data_standardized = (combined_data - combined_data.mean()) / combined_data.std()

    # Apply PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_data_standardized.T)  # Transpose to make samples the data points

    # Create a color palette for the plot
    palette = sns.color_palette("Set2", len(predictor_names) + 1)

    # Create the plot
    plt.figure(figsize=(14, 10))

    # Plot each group (results + predictors)
    labels = ['Results'] + predictor_names
    for i in range(len(labels)):
        plt.scatter(pca_result[i, 0], pca_result[i, 1],
                    label=labels[i], color=palette[i], s=100, alpha=0.85, edgecolors='k', linewidth=0.6)

    # Enhance plot aesthetics
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)", fontsize=26)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)", fontsize=26)
    plt.title('PCA of Predictors and Results', fontsize=30, fontweight='bold')
    plt.legend(title_fontsize='17', fontsize='16', loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_histogram_for_individual_correlations(df_results_ranked, predictor_dfs, predictor_names):
    for idx, df_predictor in enumerate(predictor_dfs):
        individual_correlations = []
        individuals = df_results_ranked.columns

        # For each individual, calculate Pearson correlation between the results and predictor columns
        for individual in individuals:
            corr = pearson_correlation(df_results_ranked[individual], df_predictor[individual])
            individual_correlations.append(corr)

        # Calculate correlation using all individuals
        overall_corr = pearson_correlation(df_results_ranked.values.flatten(), df_predictor.values.flatten())

        # Calculate median and average correlation
        median_corr = np.median(individual_correlations)
        avg_corr = np.mean(individual_correlations)

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.bar(individuals, individual_correlations, color='skyblue', label='Individual Correlations')
        plt.axhline(y=overall_corr, color='red', linestyle='--', label=f'Overall Correlation: {overall_corr:.2f}')
        plt.axhline(y=median_corr, color='green', linestyle='--', label=f'Median Correlation: {median_corr:.2f}')
        plt.axhline(y=avg_corr, color='orange', linestyle='--', label=f'Average Correlation: {avg_corr:.2f}')
        plt.ylim([-1, 1])  # Set y-axis range to [-1, 1] for Pearson correlation
        plt.title(f'Pearson Correlation per Individual - {predictor_names[idx]}', fontsize=18, fontweight='bold')
        plt.xlabel('Individual', fontsize=14)
        plt.ylabel('Pearson Correlation', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

def plot_predictor_relationship_with_results_heatmap(predictor_dfs, results_df, predictor_names):
    # Add results to predictor names
    predictor_names_with_results = predictor_names + ["Results"]

    # Append the results dataframe to the predictor list
    all_dfs = predictor_dfs + [results_df]

    # Calculate pairwise correlations between predictors (including results) for each individual
    num_predictors = len(all_dfs)
    correlation_matrix = np.zeros((num_predictors, num_predictors))

    for i in range(num_predictors):
        for j in range(num_predictors):
            corr = pearson_correlation(all_dfs[i].values.flatten(), all_dfs[j].values.flatten())
            correlation_matrix[i, j] = corr

    # Plot heatmap with a more engaging color palette
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=predictor_names_with_results, yticklabels=predictor_names_with_results,
                cbar_kws={'label': 'Pearson Correlation'})
    plt.title("Correlation Between Predictors and Results", fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Additional required functions
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
            perm_diff_distribution = np.abs(permuted_scores_distribution[:, i] - permuted_scores_distribution[:, j])
            p_value = np.mean(perm_diff_distribution >= actual_diff)
            p_values.append((i, j, p_value))

    return p_values, actual_scores

# File input GUI
def file_input_gui():
    layout = [
        [sg.Text('Select CSV files for each predictor (click "+" to add more):')],
        [sg.Column([[sg.Input(key='Predictor0'), sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))]],
                   key='PredictorsColumn')],
        [sg.Button('+', key='AddPredictor')],
        [sg.Checkbox('No results file (use simulation)', key='NoResults', default=False, enable_events=True)],
        [sg.Text('Select CSV file for results:', key='ResultsLabel', visible=True)],
        [sg.Input(key='Results', visible=True), sg.FileBrowse(file_types=(("CSV Files", "*.csv"),), key='ResultsBrowse', visible=True)],
        [sg.Checkbox('Data is already ranked', key='Ranked', default=False)],
        [sg.Text('Simulation Parameters:', font=('Arial', 12, 'bold'))],
        [sg.Text('Minimum correlation:', key='MinCorrLabel'),
         sg.InputText('0.7', key='MinCorrelation')],
        [sg.Text('Number of subsets to find (N):', key='NumSubsetsLabel'),
         sg.InputText('1', key='NumSubsets')],
        [sg.Text('Enter number of iterations:', key='NumIterationsLabel', visible=False),
         sg.InputText('1', key='NumIterations', visible=False)],
        [sg.Text('All files must be CSV.', font=('Arial', 10, 'italic'))],
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
            window.extend_layout(window['PredictorsColumn'],
                                 [[sg.Input(key=new_key), sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))]])
            window.refresh()

        if event == 'NoResults':
            if values['NoResults']:
                # Hide results file selection
                window['ResultsLabel'].update(visible=False)
                window['Results'].update(visible=False)
                window['ResultsBrowse'].update(visible=False)
                # Show number of iterations
                window['NumIterationsLabel'].update(visible=True)
                window['NumIterations'].update(visible=True)
            else:
                # Show results file selection
                window['ResultsLabel'].update(visible=True)
                window['Results'].update(visible=True)
                window['ResultsBrowse'].update(visible=True)
                # Hide number of iterations
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

            min_correlation = values['MinCorrelation']
            num_subsets = values['NumSubsets']

            if not min_correlation or not num_subsets:
                sg.popup("Please enter minimum correlation and number of subsets.", title="Error")
                continue

            min_correlation = float(min_correlation)
            num_subsets = int(num_subsets)

            if no_results:
                num_iterations = values['NumIterations']
                if not num_iterations:
                    sg.popup("Please enter number of iterations.", title="Error")
                    continue
                num_iterations = int(num_iterations)
                results_file = None
            else:
                results_file = values['Results']
                if not results_file:
                    sg.popup("Please select a results file.", title="Error")
                    continue
                num_iterations = None

            window.close()
            return predictor_files, results_file, is_ranked, no_results, min_correlation, num_subsets, num_iterations

def main():
    predictor_files, results_file, is_ranked, no_results, min_correlation, num_subsets, num_iterations = file_input_gui()
    if not predictor_files:
        print("Operation cancelled or no files selected.")
        return

    try:
        print("Files were uploaded and are being processed...")

        # Preprocess data to get common drugs and individuals
        data_dict, common_drugs, common_individuals, df_results_ranked, ranked_predictor_dfs, predictor_names = preprocess_data(
            predictor_files, results_file, is_ranked, no_results)

        num_predictors = len(predictor_names)

        if no_results:
            # Handle the simulation case
            # Use the number of drugs and individuals from the intersected data
            num_drugs = len(common_drugs)
            num_individuals = len(common_individuals)

            print(f"The simulation will run on {num_drugs} drugs and {num_individuals} individuals with {num_predictors} predictors.")

            correlations_matrix, num_drugs_first, num_individuals_first, subsets_df = run_project_simulation(
                num_individuals, num_drugs, num_predictors, min_correlation, num_subsets, common_drugs, common_individuals, predictor_names, num_iterations)

            if correlations_matrix is not None:
                # Plot static heatmaps
                plot_static_heatmap(correlations_matrix, min_correlation, num_drugs_first, num_individuals_first, subsets_df, num_subsets)

        else:
            # Original code processing results
            num_drugs = len(common_drugs)
            num_individuals = len(common_individuals)
            data_dict["Results"] = df_results_ranked.values.tolist()
            num_predictors += 1  # Including the results

            print("Calculating distances for each predictor...")
            p_values, full_data_scores = permutation_test(df_results_ranked, ranked_predictor_dfs)
            print("Scoring complete. Plotting bar chart...")
            plot_bar_chart_with_significance(full_data_scores, predictor_names, p_values)

            # Plot the histogram for each predictor
            plot_histogram_for_individual_correlations(df_results_ranked, ranked_predictor_dfs, predictor_names)
            apply_pca_and_plot(df_results_ranked, ranked_predictor_dfs, predictor_names, common_individuals)
            plot_predictor_relationship_with_results_heatmap(ranked_predictor_dfs, df_results_ranked, predictor_names)

            # Optimize ranking correlation with drugs by highest std and individuals arbitrarily
            print("Optimizing ranking correlation with drugs by highest std and individuals arbitrarily...")
            correlations, num_drugs_first, num_individuals_first, subsets_df = optimize_ranking_correlation(
                df_results_ranked, ranked_predictor_dfs, min_correlation, num_subsets)

            # Plot static heatmap
            print("Plotting static heatmap...")
            plot_static_heatmap(correlations, min_correlation, num_drugs_first, num_individuals_first, subsets_df, num_subsets)

            print("Processing completed successfully.")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
