import ast
import re

import numpy as np
import pandas as pd
import pingouin as pg
from utils.ComputationalModeling import parameter_extractor, trialwise_extractor

# read behavioral data
E1_data = pd.read_csv('./data/E1_data_with_assignments.csv')
E2_data = pd.read_csv('./data/E2_data_full.csv')
E2_summary = pd.read_csv('./data/E2_summary_full.csv')

print(E2_summary.columns)
print(E1_data.groupby('Condition')['Sex'].value_counts())

# Rename E2 columns to match E1
E2_data.rename(columns={'Gender': 'Sex','BISScore': 'Bis11Score', 'CESDScore': 'CESD',
                        'ESIBF_disinhScore': 'ESIBF_Disinhibition', 'ESIBF_sScore': 'ESIBF_SubstanceUse',
                        'ESIBF_aggreScore': 'ESIBF_Aggression', 'PSWQScore': 'PSWQ', 'STAI_Score': 'STAIS',
                        'STAI_TScore': 'STAIT'}, inplace=True)
subj_level_col = ['Subnum', 'Age', 'Sex', 'Race', 'Ethnicity', 'Big5O', 'Big5C', 'Big5E', 'Big5A', 'Big5N',
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
            mean_bic = df['BIC'].mean()
            print(f"{name}: {mean_bic}")

print_mean_bic(E1_results, "E1")
print_mean_bic(E2_all_results, "E2 All")
print_mean_bic(E2_baseline_results, "E2 Baseline")
print_mean_bic(E2_freq_results, "E2 Frequency")

# ======================================================================================================================
# Best model
# ======================================================================================================================
model_baseline_nlls = []
for model_name, df in E2_baseline_results.items():
    model_df = df[['Subnum', 'BIC']].copy()
    model_df['model'] = model_name
    model_df['Condition'] = 'Baseline'
    model_baseline_nlls.append(model_df)

model_frequency_nlls = []
for model_name, df in E2_freq_results.items():
    model_df = df[['Subnum', 'BIC']].copy()
    model_df['model'] = model_name
    model_df['Condition'] = 'Frequency'
    model_frequency_nlls.append(model_df)

# Step 2: Concatenate all models
all_nlls = pd.concat(model_baseline_nlls + model_frequency_nlls, ignore_index=True)

# Step 3: Find the best model (lowest nll) per participant
best_model_per_participant = (
    all_nlls.sort_values('BIC')
            .drop_duplicates(['Subnum', 'Condition'], keep='first')
            .reset_index(drop=True)
)
# Assuming 'group' is in any of the model DataFrames, like decay
group_info = E2_baseline_results['delta'][['Subnum', 'group_baseline']]
all_nlls = all_nlls.merge(group_info, on='Subnum', how='left')
best_model_per_participant = best_model_per_participant.merge(group_info, on='Subnum', how='left')

model_counts = (best_model_per_participant.groupby(['group_baseline', 'Condition', 'model']).size().unstack(fill_value=0))

avg_bic = (all_nlls.groupby(['group_baseline', 'Condition', 'model'])['BIC'].mean().unstack(level='model').round(2))

print(avg_bic)

print(model_counts)

# ======================================================================================================================
# Extract the dual process model for model analysis
# ======================================================================================================================
model = 'dual'
# param_name = ['t', 'alpha', 'w', 'lambda']
param_name = ['t', 'alpha', 'subj_weight']

E2_all_model = E2_all_results[model].copy()
E2_baseline_model = E2_baseline_results[model].copy()
E2_freq_model = E2_freq_results[model].copy()

E2_all_model = parameter_extractor(E2_all_model, param_name)
E2_baseline_model = parameter_extractor(E2_baseline_model, param_name)
E2_freq_model = parameter_extractor(E2_freq_model, param_name)

# combine them and add a column for the condition
E2_baseline_model['Condition'] = 'Baseline'
E2_freq_model['Condition'] = 'Frequency'
E2_all_model['Condition'] = 'All'
E2_dual = pd.concat([E2_baseline_model, E2_freq_model], ignore_index=True).sort_values(by=['Subnum']). reset_index(drop=True)
# remove group 2
E2_group13 = E2_dual[E2_dual['group_baseline'] != 2].copy()

# Mixed ANOVA for dual process model
var = 'subj_weight'  # the variable to analyze
mixed_anova_dual = pg.mixed_anova(E2_dual, dv=var, within='Condition', between='group_baseline', subject='Subnum')
print(E2_dual.groupby(['group_baseline', 'Condition'])[var].mean().reset_index())