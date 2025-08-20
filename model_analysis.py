import numpy as np
import pandas as pd
import pingouin as pg
import itertools
import functools
import scipy.stats as stats
from utils.ComputationalModeling import (parameter_extractor, trialwise_extractor, parse_np_list_str, weight_calculation,
                                         bayes_factor, get_best_df, vb_model_selection, compute_exceedance_prob)

# read behavioral data
E1_data = pd.read_csv('./data/E1_data_full.csv')
E1_summary = pd.read_csv('./data/E1_summary_full.csv')
E2_data = pd.read_csv('./data/E2_data_full.csv')
E2_summary = pd.read_csv('./data/E2_summary_full.csv')
E1_inattentive = pd.read_csv('./data/E1_inattentive_participants.csv')
E2_inattentive = pd.read_csv('./data/E2_inattentive_participants.csv')

print(E2_data.columns)
print(E1_data.groupby('Condition')['Sex'].value_counts())

# Rename E2 columns to match E1
E2_data.rename(columns={'Gender': 'Sex','BISScore': 'Bis11Score', 'CESDScore': 'CESD',
                        'ESIBF_disinhScore': 'ESIBF_Disinhibition', 'ESIBF_sScore': 'ESIBF_SubstanceUse',
                        'ESIBF_aggreScore': 'ESIBF_Aggression', 'PSWQScore': 'PSWQ', 'STAISScore': 'STAIS',
                        'STAITScore': 'STAIT'}, inplace=True)
subj_level_col = ['Subnum', 'Age', 'Sex', 'Race', 'Ethnicity', 'Big5O', 'Big5C', 'Big5E', 'Big5A', 'Big5N',
                  'Bis11Score', 'CESD', 'ESIBF_Aggression', 'ESIBF_Disinhibition', 'ESIBF_SubstanceUse', 'NPI', 'PSWQ',
                  'STAIS', 'STAIT', 'TPM_Boldness', 'TPM_Disinhibition', 'TPM_Meanness']
personality_col = ['Subnum', 'Big5O', 'Big5C', 'Big5E', 'Big5A', 'Big5N']
psychiatric_col = ['Subnum', 'Bis11Score', 'CESD', 'ESIBF_Aggression', 'ESIBF_Disinhibition', 'ESIBF_SubstanceUse', 'NPI', 'PSWQ',
                  'STAIS', 'STAIT', 'TPM_Boldness', 'TPM_Disinhibition', 'TPM_Meanness']
E1_subj_level_col = subj_level_col + ['prob1', 'prob2', 'prob3', 'assignments']
E2_subj_level_col = subj_level_col + ['prob1_baseline', 'prob2_baseline', 'prob3_baseline', 'group_baseline',
                                      'prob1_frequency', 'prob2_frequency', 'prob3_frequency', 'group_frequency']

# Select only the subject-level columns if they exist in the data
E1_subj_data = E1_data[[col for col in E1_subj_level_col if col in E1_data.columns]].drop_duplicates()
E2_subj_data = E2_data[[col for col in E2_subj_level_col if col in E2_data.columns]].drop_duplicates()

# Create a trial column
E1_data['trial'] = E1_data.groupby('Subnum').cumcount() + 1
E2_data['trial'] = E2_data.groupby(['Subnum', 'Condition']).cumcount() + 1

# read the model fitting results
# result_names = ['delta', 'delta_PVL', 'delta_PVL_relative', 'decay', 'decay_PVL',
#                 'decay_PVL_relative', 'delta_decay', 'delta_decay_PVL', 'delta_decay_PVL_relative', 'dual']
result_names = ['delta', 'delta_PVL', 'delta_asymmetric', 'decay', 'decay_PVL', 'decay_win', 'delta_asymmetric_decay_win']
E1_results_dir = './data/ModelFitting/E1/'
E2_all_results_dir = './data/ModelFitting/E2/'
E2_baseline_results_dir = './data/ModelFitting/E2/Baseline/'
E2_freq_results_dir = './data/ModelFitting/E2/Frequency/'

# read all results
def read_results(directory, result_names, raw_data, inattentive):
    results = {}
    for name in result_names:
        file_path = f'{directory}{name}_results.csv'
        try:
            results[name] = pd.read_csv(file_path)
            results[name]['Model'] = name  # Add a column for the model name
            results[name].rename(columns={'participant_id': 'Subnum'}, inplace=True)
            results[name] = results[name].merge(raw_data, on='Subnum', how='left')
            # remove inattentive participants
            results[name] = results[name][~results[name]['Subnum'].isin(inattentive['Subnum'].unique())].reset_index(drop=True)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
    return results

E1_results = read_results(E1_results_dir, result_names, E1_subj_data, E1_inattentive)
E2_all_results = read_results(E2_all_results_dir, result_names, E2_subj_data, E2_inattentive)
E2_baseline_results = read_results(E2_baseline_results_dir, result_names, E2_subj_data, E2_inattentive)
E2_freq_results = read_results(E2_freq_results_dir, result_names, E2_subj_data, E2_inattentive)

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
parameter_map = {
    'delta': ['t', 'alpha'],
    'delta_PVL': ['t', 'alpha', 'scale', 'la'],
    'delta_PVL_relative': ['t', 'alpha', 'scale', 'la'],
    'decay': ['t', 'alpha'],
    'decay_PVL': ['t', 'alpha', 'scale', 'la'],
    'decay_PVL_relative': ['t', 'alpha', 'scale', 'la'],
    'dual': ['t', 'alpha', 'subj_weight'],
    'delta_decay': ['t', 'alpha', 'weight'],
    'delta_decay_PVL': ['t', 'alpha', 'scale', 'la', 'weight'],
    'delta_decay_PVL_relative': ['t', 'alpha', 'scale', 'la', 'weight'],
    'delta_asymmetric': ['t', 'alpha', 'alpha_neg'],
    'delta_asymmetric_decay_win': ['t', 'alpha', 'alpha_neg', 'weight'],
    'decay_win': ['t', 'alpha'],
    'delta_decay_win': ['t', 'alpha', 'weight'],
    'delta_decay_PVL_win': ['t', 'alpha', 'scale', 'la', 'weight'],
    'delta_decay_PVL_relative_win': ['t', 'alpha', 'scale', 'la', 'weight'],
}

model_baseline_bics = []
for model_name, df in E2_baseline_results.items():
    model_df = df[['Subnum', 'best_parameters', 'AIC', 'BIC']].copy()
    model_df['model'] = model_name
    model_df['Condition'] = 'Baseline'
    model_df = parameter_extractor(model_df, parameter_map[model_name])
    model_baseline_bics.append(model_df)

model_frequency_bics = []
for model_name, df in E2_freq_results.items():
    model_df = df[['Subnum', 'best_parameters', 'AIC', 'BIC']].copy()
    model_df['model'] = model_name
    model_df['Condition'] = 'Frequency'
    model_df = parameter_extractor(model_df, parameter_map[model_name])
    model_frequency_bics.append(model_df)

# Concatenate all models
all_bics = pd.concat(model_baseline_bics + model_frequency_bics, ignore_index=True)

# Find the best model per participant
best_model_per_participant = (
    all_bics.sort_values('BIC')
            .drop_duplicates(['Subnum', 'Condition'], keep='first')
            .reset_index(drop=True)
)
# Assuming 'group' is in any of the model DataFrames, like decay
group_info = E2_baseline_results['delta'][['Subnum', 'group_baseline']]
all_bics = all_bics.merge(group_info, on='Subnum', how='left')
best_model_per_participant = best_model_per_participant.merge(group_info, on='Subnum', how='left')

model_counts = (best_model_per_participant.groupby(['Condition', 'model']).size().unstack(fill_value=0)).T

avg_bic = (all_bics.groupby(['group_baseline', 'Condition', 'model'])['BIC'].mean().unstack(level='model').round(2))

print(avg_bic)

print(model_counts)

# Combine all BIC values into one DataFrame for summary
model_summary = pd.concat(model_baseline_bics + model_frequency_bics, ignore_index=True)
model_summary = model_summary.sort_values(['Subnum', 'Condition'], ignore_index=True)
model_summary.to_csv('./data/model_summary.csv', index=False)

# # Combine all BIC values into the original summary DataFrame
# for i, df in enumerate(model_frequency_bics):
#     E2_summary = pd.merge(E2_summary, df[['Subnum', 'Condition', 'model', 'BIC']],
#                           on=['Subnum', 'Condition'], how='outer', suffixes=('', f'_{df["model"].iloc[0]}'))
#
# for i, df in enumerate(model_baseline_bics):
#     E2_summary = pd.merge(E2_summary, df[['Subnum', 'Condition', 'BIC']],
#                           on=['Subnum', 'Condition'], how='left', suffixes=('', f'_{df["model"].iloc[0]}'))

summary_list = []

# iterate through all dataframes in both lists
for df in itertools.chain(model_frequency_bics, model_baseline_bics):
    model_name = df['model'].iloc[0]

    # keep only numeric columns after dropping id-like cols
    cols_to_drop = ['Subnum', 'Condition', 'model']
    numeric_means = df.drop(columns=cols_to_drop, errors='ignore').select_dtypes('number').mean()

    # attach identifiers
    numeric_means['model'] = model_name
    if 'Condition' in df.columns:
        numeric_means['Condition'] = df['Condition'].iloc[0]

    summary_list.append(numeric_means)

summary_df = pd.DataFrame(summary_list)
summary_df['model'] = pd.Categorical(summary_df['model'], categories=result_names, ordered=True)
summary_df = summary_df.sort_values(['Condition', 'model']).reset_index(drop=True)
summary_df = summary_df.groupby('Condition', group_keys=False).apply(weight_calculation, ['BIC'])
summary_df = summary_df.round(3)
summary_df.to_csv('./data/model_summary_statistics.csv', index=False)
print(summary_df)

# Calculate Bayes Factor for model comparison
best_model = 'delta_asymmetric_decay_win'
bayes_factor_results = []

for df in model_frequency_bics:
    model_name = df['model'].iloc[0]
    cond = df['Condition'].iloc[0]
    best_df = get_best_df(model_frequency_bics, best_model, cond)

    BF = bayes_factor(null_results=df, alternative_results=best_df)  # returns a float

    bayes_factor_results.append({
        'Condition': cond,
        'model': model_name,
        f'BF_vs_{best_model}': BF
    })

bf_table = pd.DataFrame(bayes_factor_results).sort_values(['Condition', 'model']).reset_index(drop=True).round(3)
print(bf_table)

# Now calculate variational Bayes indices
K = 7 # number of models

# select columns that end with BIC
dfs = []
for df in model_baseline_bics:
    model_name = df['model'].iloc[0]
    cond = df['Condition'].iloc[0]
    tmp = df[['Subnum', 'Condition', 'BIC']].copy()
    tmp = tmp.rename(columns={'BIC': f'{model_name}_BIC'})
    dfs.append(tmp)

combined_BICs = functools.reduce(lambda left, right: pd.merge(left, right, on=['Subnum','Condition'], how='outer'),dfs)
bic_cols = [col for col in combined_BICs.columns if col.endswith('BIC')]
log_evidences = combined_BICs[bic_cols].values / (-2)

# Run VB model selection
alpha0 = np.ones(K)  # uniform prior
alpha_est, g_est = vb_model_selection(log_evidences, alpha0=alpha0, tol=1e-12, max_iter=50000)

# calculate the exceedance probabilities
ex_probs = compute_exceedance_prob(alpha_est, n_samples=100000)

# convert all to DataFrame for better readability
alpha_est_df = pd.DataFrame(alpha_est, index=bic_cols).round(3)
model_freq = pd.DataFrame((alpha_est / np.sum(alpha_est)).round(3), index=bic_cols, columns=['Frequency'])
ex_probs_df = pd.DataFrame(ex_probs.round(3), index=bic_cols, columns=['Exceedance Probability'])

print("Final alpha (Dirichlet parameters):", alpha_est_df.round(3))
# print("Posterior model probabilities per subject:\n", g_est)
print("Expected model frequencies:", model_freq.round(3))
print("Exceedance probabilities:", ex_probs_df.round(3))

# ======================================================================================================================
# E2 - Extract the dual process model for model analysis
# ======================================================================================================================
model      = "dual"
param_name = ["t", "alpha", "subj_weight"]
# param_name = ['t', 'alpha', 'w', 'lambda']
var_name   = ["best_weight", "best_obj_weight"]

sources = {
    "Baseline" : E2_baseline_results,
    "Frequency": E2_freq_results,
    "All"      : E2_all_results,
}

wide_dfs = []
for cond, results in sources.items():
    # Extract the model parameters
    df = results[model].copy()
    df = parameter_extractor(df, param_name)
    df = df[["Subnum"] + param_name + var_name]

    # Explode trial‐wise variables
    for col in var_name:
        df[col] = df[col].apply(parse_np_list_str)
    df = df.explode(var_name).reset_index(drop=True)

    # Rename the columns to include the condition and model
    if cond == "All":
        suffix = f"_{cond}"
        rename_map = {p: p + suffix for p in (param_name + var_name)}
        df = df.rename(columns=rename_map)
    else:
        suffix = ''

    df["Condition"] = cond

    # Insert the first trial for each subject because the first trial is not modeled
    param_cols = [p + suffix for p in param_name]
    var_cols = [v + suffix for v in var_name]

    first = (
        df.groupby("Subnum")[param_cols]
        .first()
        .reset_index()
    )
    for vcol in var_cols:
        first[vcol] = np.nan
    first["Condition"] = cond

    # --- 5) stitch them together & sort
    df = pd.concat([first, df], ignore_index=True)
    df = df.sort_values('Subnum').reset_index(drop=True)

    # (optional) drop the trial column if you don’t need it downstream
    # df = df.drop(columns="trial")

    wide_dfs.append(df)

# Combine baseline and frequency into a long format
E2_conditionwise = pd.merge(wide_dfs[0], wide_dfs[1],
                            on=['Subnum', 'Condition', 't', 'alpha', 'subj_weight', 'best_weight', 'best_obj_weight'],
                            how='outer', suffixes=('_baseline', '_frequency'))
E2_conditionwise['trial'] = E2_conditionwise.groupby(['Subnum', 'Condition']).cumcount() + 1

# Combine the data with the subject-level data
E2_data_merged = pd.merge(E2_conditionwise, E2_data, on=['Subnum', 'Condition', 'trial'], how='outer')
E2_data_merged = pd.concat([E2_data_merged, wide_dfs[2][['t_All', 'alpha_All', 'subj_weight_All', 'best_weight_All', 'best_obj_weight_All']]], axis=1).reset_index(drop=True)
E2_data_merged[E2_data_merged['TrialType'].isin(['CA', 'CB', 'AD', 'BD'])].to_csv('./data/E2_data_testing_modeled.csv', index=False)
E2_data_merged.to_csv('./data/E2_data_modeled.csv', index=False)

E2_summary = pd.merge(E2_summary, E2_conditionwise[['Subnum', 'Condition', 't', 'alpha', 'subj_weight']].drop_duplicates(),
                        on=['Subnum', 'Condition'], how='left')
E2_summary = pd.merge(E2_summary, wide_dfs[2][['Subnum', 't_All', 'alpha_All', 'subj_weight_All']].drop_duplicates(),
                        on='Subnum', how='left')
E2_summary.to_csv('./data/E2_summary_modeled.csv', index=False)

# Correlate the condition-wise parameters with all-condition parameters
for param in param_name:
    baseline_param = wide_dfs[0][['Subnum', param]].drop_duplicates()
    frequency_param = wide_dfs[1][['Subnum', param]].drop_duplicates()
    all_param = wide_dfs[2][['Subnum', f"{param}_All"]].drop_duplicates()

    print(f'Correlation between {param} in Baseline and All:')
    print(stats.pearsonr(baseline_param[param], all_param[f"{param}_All"]))
    print(f'Correlation between {param} in Frequency and All:')
    print(stats.pearsonr(frequency_param[param], all_param[f"{param}_All"]))
    print(f'Correlation between {param} in Baseline and Frequency:')
    print(stats.pearsonr(baseline_param[param], frequency_param[param]))

# Extract individual differences parameters for E2
accuracy_wide = E2_summary.pivot(index=['Subnum', 'Condition'], columns='TrialType', values='BestOption').reset_index()
# reorder the columns to match the original order
trial_types = ['AB', 'CD', 'CA', 'CB', 'AD', 'BD']
accuracy_wide = accuracy_wide[['Subnum', 'Condition'] + trial_types]
print(accuracy_wide.columns)
accuracy_wide[accuracy_wide['Condition'] == 'Baseline'].drop(columns='Condition').to_csv('./data/E2_summary_CA_baseline.csv', index=False)
accuracy_wide[accuracy_wide['Condition'] == 'Frequency'].drop(columns='Condition').to_csv('./data/E2_summary_CA_frequency.csv', index=False)
subj_data = E2_data[[col for col in psychiatric_col if col in E2_data]].drop_duplicates().to_csv('./data/E2_subj_data.csv', index=False)
print(E2_data[[col for col in psychiatric_col if col in E2_data]].describe())

# ----------------------------------------------------------------------------------------------------------------------
# Extract model fit parameters for E2
# ----------------------------------------------------------------------------------------------------------------------

# # ======================================================================================================================
# # E1 - Extract the dual process model for model analysis (Unused)
# # ======================================================================================================================
# model      = "dual"
# param_name = ["t", "alpha", "subj_weight"]
# var_name   = ["best_weight", "best_obj_weight"]
#
# E1_dual = E1_results[model].copy()
# E1_dual = parameter_extractor(E1_dual, param_name)
# E1_dual = E1_dual[["Subnum"] + param_name + var_name]
#
# # Explode trial‐wise variables
# for col in var_name:
#     E1_dual[col] = E1_dual[col].apply(parse_np_list_str)
# E1_dual = E1_dual.explode(var_name).reset_index(drop=True)
#
# # Insert the first trial for each subject because the first trial is not modeled
# first = (E1_dual.groupby("Subnum")[param_name]
#     .first()
#     .reset_index()
# )
# for vcol in var_name:
#     first[vcol] = np.nan
#
# E1_dual = pd.concat([first, E1_dual], ignore_index=True)
# E1_dual = E1_dual.sort_values('Subnum').reset_index(drop=True)
# E1_dual['trial'] = E1_dual.groupby('Subnum').cumcount() + 1
#
# # Combine the data with the subject-level data
# E1_data_merged = pd.merge(E1_dual, E1_data, on=['Subnum', 'trial'], how='outer')
# E1_data_merged = E1_data_merged.sort_values('Subnum').reset_index(drop=True)
# E1_data_merged.rename(columns={'SetSeen.': 'TrialType'}, inplace=True)
# E1_data_merged['TrialType'] = E1_data_merged['TrialType'].replace({0: 'AB', 1: 'CD', 2: 'CA', 3: 'CB', 4: 'AD', 5: 'BD'})
# E1_data_merged[E1_data_merged['TrialType'].isin(['CA', 'CB', 'AD', 'BD'])].to_csv('./data/E1_data_testing_modeled.csv', index=False)
# E1_data_merged.to_csv('./data/E1_data_modeled.csv', index=False)
# E1_summary = pd.merge(E1_summary, E1_dual[['Subnum', 't', 'alpha', 'subj_weight']].drop_duplicates(),
#                       on='Subnum', how='left')
# E1_summary.to_csv('./data/E1_summary_modeled.csv', index=False)
