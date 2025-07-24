import numpy as np
import pandas as pd
from utils.ComputationalModeling import parameter_extractor

# read behavioral data
E1_data = pd.read_csv('./data/E1_data_with_assignments.csv')
E2_data = pd.read_csv('./data/E2_data_with_assignments.csv')
print(E1_data.columns)
print(E1_data.groupby('Condition')['Sex'].value_counts())

# Rename E2 columns to match E1
E2_data.rename(columns={'Gender': 'Sex','BISScore': 'Bis11Score', 'CESDScore': 'CESD',
                        'ESIBF_disinhScore': 'ESIBF_Disinhibition', 'ESIBF_sScore': 'ESIBF_SubstanceUse',
                        'ESIBF_aggreScore': 'ESIBF_Aggression', 'PSWQScore': 'PSWQ', 'STAI_Score': 'STAIS',
                        'STAI_TScore': 'STAIT'}, inplace=True)
subj_level_col = ['Subnum', 'Condition', 'Age', 'Sex', 'Race', 'Ethnicity', 'Big5O', 'Big5C', 'Big5E', 'Big5A', 'Big5N',
                     'Bis11Score', 'CESD', 'ESIBF_Disinhibition', 'ESIBF_SubstanceUse', 'NPI', 'PSWQ', 'STAIS', 'STAIT',
                     'TPM_Boldness', 'TPM_Disinhibition', 'TPM_Meanness']
E1_subj_level_col = subj_level_col + ['prob1', 'prob2', 'prob3', 'assignments']
E2_subj_level_col = subj_level_col + ['prob1_baseline', 'prob2_baseline', 'prob3_baseline', 'group_baseline',
                                      'prob1_frequency', 'prob2_frequency', 'prob3_frequency', 'group_frequency']

# Select only the subject-level columns if they exist in the data
E1_subj_data = E1_data[[col for col in E1_subj_level_col if col in E1_data.columns]].drop_duplicates()
E2_subj_data = E2_data[[col for col in E2_subj_level_col if col in E2_data.columns]].drop_duplicates()

# read the model fitting results
result_names = ['delta', 'delta_PVL', 'delta_PVL_relative', 'decay', 'decay_win', 'decay_PVL',
                'decay_PVL_relative', 'decay_PVPE', 'dual', 'dual_sensitivity']
E1_results_dir = './data/ModelFitting/E1/'
E2_all_results_dir = './data/ModelFitting/E2/'
E2_baseline_results_dir = './data/ModelFitting/E2/Baseline/'
E2_freq_results_dir = './data/ModelFitting/E2/Frequency/'

# read all results
def read_results(directory, result_names, raw_data):
    results = {}
    for name in result_names:
        file_path = f'{directory}{name}_results.csv'
        try:
            results[name] = pd.read_csv(file_path)
            results[name]['Model'] = name  # Add a column for the model name
            results[name].rename(columns={'participant_id': 'Subnum'}, inplace=True)
            results[name] = results[name].merge(raw_data, on='Subnum', how='left')
        except FileNotFoundError:
            print(f"File {file_path} not found.")
    return results

E1_results = read_results(E1_results_dir, result_names, E1_subj_data)
E2_all_results = read_results(E2_all_results_dir, result_names, E2_subj_data)
E2_baseline_results = read_results(E2_baseline_results_dir, result_names, E2_subj_data)
E2_freq_results = read_results(E2_freq_results_dir, result_names, E2_subj_data)

# print the mean BIC per model per condition in E1
def print_mean_bic(results, condition):
    print(f"Mean BIC for {condition}:")
    for name, df in results.items():
        if not df.empty:
            mean_bic = df.groupby('Condition')['BIC'].mean()
            print(f"{name}: {mean_bic}")

print_mean_bic(E1_results, "E1")
print_mean_bic(E2_all_results, "E2 All")
print_mean_bic(E2_baseline_results, "E2 Baseline")
print_mean_bic(E2_freq_results, "E2 Frequency")

# ======================================================================================================================
# Extract the dual process model for model analysis
# ======================================================================================================================
E2_all_dual = E2_all_results['dual'].copy()
E2_baseline_dual = E2_baseline_results['dual'].copy()
E2_freq_dual = E2_freq_results['dual'].copy()

E2_all_dual = parameter_extractor(E2_all_dual)
E2_baseline_dual = parameter_extractor(E2_baseline_dual)
E2_freq_dual = parameter_extractor(E2_freq_dual)

print(E2_freq_dual.groupby('group_baseline')['subj_weight'].mean())