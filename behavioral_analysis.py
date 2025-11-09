import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from dask.array.stats import ttest_rel
from scipy.stats import ttest_1samp
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

# ======================================================================================================================
# E2
# ======================================================================================================================
# Read in the E2_summary
E2_data = pd.read_csv('./data/E2_data_with_assignments.csv')
E2_summary = pd.read_csv('./data/E2_summary_with_assignments.csv')
E2_data['Phase'] = (E2_data.groupby(['Subnum', 'Condition']).cumcount() // 20) + 1
print(E2_data.columns)
# print E2_summary CA with best option > 0.4 < 0.6
print(E2_summary[(E2_summary['TrialType'] == 'CA') & (E2_summary['BestOption'] > 0.40) & (E2_summary['BestOption'] < 0.60)].groupby('Condition').size())  #



# Rename the trial type
E2_training = E2_data[E2_data['TrialType'].isin(['AB', 'CD'])].copy()
E2_training_summary = E2_summary[E2_summary['TrialType'].isin(['AB', 'CD'])].copy()

# calculate overall training accuracy
weight_map = {
    ('Baseline', 'AB'):       0.5,
    ('Baseline', 'CD'):       0.5,
    ('Frequency', 'AB'):      2/3,
    ('Frequency', 'CD'):      1/3,
}
E2_training_summary['weight'] = E2_training_summary.apply(lambda row: weight_map[(row.Condition, row.TrialType)], axis=1)
E2_training_summary['weighted_acc'] = E2_training_summary['BestOption'] * E2_training_summary['weight']
training_acc = (E2_training_summary.groupby(['Subnum', 'Condition'])['weighted_acc'].sum().reset_index(name='training_accuracy'))

E2_training_summary = E2_training_summary.merge(training_acc, on=['Subnum', 'Condition'], how='left')
E2_training_summary.drop(columns=['weight', 'weighted_acc'], inplace=True)

# median split for training accuracy
E2_training_summary['group_training'] = (E2_training_summary.groupby('Condition')['training_accuracy']
                                         .transform(lambda x: np.where(x >= x.quantile(0.75), 'High',
                                                                       np.where(x <= x.quantile(0.25), 'Low', 'Average'))))
group_training = E2_training_summary[['Subnum', 'Condition', 'training_accuracy', 'group_training']].drop_duplicates()

# unstack the group_training DataFrame to create a wide format
wide = group_training.set_index(['Subnum', 'Condition'])[['training_accuracy', 'group_training']].unstack('Condition')
wide.reset_index(inplace=True)
wide.columns = ['Subnum', 'training_acc_baseline', 'training_acc_frequency', 'group_training_baseline', 'group_training_frequency']

# Merge the wide DataFrame with E2_summary
E2_summary = E2_summary.merge(group_training, on=['Subnum', 'Condition'], how='left')
E2_summary = E2_summary.merge(wide, on='Subnum', how='left')
E2_summary['group_training'] = pd.Categorical(E2_summary['group_training'], categories=['High', 'Average', 'Low'], ordered=True)

# how many participants are high in both, low in both, high in baseline and low in frequency, low in baseline and high in frequency
comb_counts = pd.crosstab(wide['group_training_baseline'], wide['group_training_frequency'],
                          rownames=['Baseline Training'],
                          colnames=['Frequency Training'])
print(comb_counts)

# Extract the difference of C choice rate across conditions
E2_summary_CA = E2_summary[E2_summary['TrialType'] == 'CA']
c_diff = (E2_summary_CA[E2_summary_CA['Condition'] == 'Baseline']['BestOption'].values -
          E2_summary_CA[E2_summary_CA['Condition'] == 'Frequency']['BestOption'].values)
E2_summary_CA['C_diff'] = np.repeat(c_diff, 2)

added_info = E2_summary_CA[['Subnum', 'Condition', 'training_accuracy', 'group_training', 'training_acc_baseline',
                            'training_acc_frequency', 'group_training_baseline', 'group_training_frequency', 'C_diff']].drop_duplicates()

# Merge the added information into all df
on = ['Subnum', 'Condition']
E2_data = pd.merge(E2_data, added_info[[c for c in added_info.columns if c not in E2_data.columns or c in on]], on=on, how='left')
E2_summary = pd.merge(E2_summary, added_info[[c for c in added_info.columns if c not in E2_summary.columns or c in on]], on=on, how='left')

E2_summary_testing = E2_summary[E2_summary['TrialType'].isin(['CA', 'CB', 'AD', 'BD'])].copy()
E2_data_testing = E2_data[E2_data['TrialType'].isin(['CA', 'CB', 'AD', 'BD'])].copy()

# Columns to remove
print(E2_summary.columns)
E2_summary_final = E2_summary.copy()
cols_to_remove = ['prob1_baseline', 'prob2_baseline', 'prob3_baseline', 'group_baseline', 'prob1_frequency',
                  'prob2_frequency', 'prob3_frequency', 'group_frequency', 'group_training', 'group_training_baseline',
                  'group_training_frequency', 'C_diff']
E2_summary_final.drop(columns=cols_to_remove, inplace=True)

# Save data
# E2_summary.to_csv('./data/E2_summary_full.csv', index=False)
# E2_summary_testing.to_csv('./data/E2_summary_testing.csv', index=False)
# E2_summary_CA.to_csv('./data/E2_summary_CA.csv', index=False)
E2_summary_final.to_csv('./data/E2_summary_final.csv', index=False)
# E2_data.to_csv('./data/E2_data_full.csv', index=False)
# E2_data_testing.to_csv('./data/E2_data_testing.csv', index=False)

# ======================================================================================================================
# Data Analysis
# ======================================================================================================================
# Demographics
print(E2_data.groupby('Subnum')['Gender'].first().value_counts()) # Gender
print(E2_data.groupby('Subnum')['Age'].first().describe()) # Age
print(E2_data.groupby('Subnum')['Race'].first().value_counts()) # Race
print(E2_data.groupby('Subnum')['Ethnicity'].first().value_counts()) #
print(E2_data.groupby('Subnum')['order'].first().value_counts()) # Education

# ANOVA for 2 TrialTypes × 6 Blocks × 2 Conditions
E2_training_blocked = E2_data[E2_data['TrialType'].isin(['AB', 'CD'])].groupby(['Subnum', 'Condition', 'TrialType', 'Phase'])['BestOption'].mean().reset_index()
# anova_3way = AnovaRM(data=E2_training_blocked, depvar='BestOption', subject='Subnum', within=['Condition', 'TrialType', 'Phase']).fit()

# Order effects
E2_order_data = E2_data.groupby(['Subnum', 'Condition', 'TrialType', 'order'])['BestOption'].mean().reset_index()

# for each trial type in each condition, compare the two orders
order_results = []
for (trial_type, condition), subdf in E2_order_data.groupby(['TrialType', 'Condition']):
    t, p = stats.ttest_ind(subdf[subdf['order'] == 'BF']['BestOption'],
                           subdf[subdf['order'] == 'FB']['BestOption'])
    order_results.append({'TrialType': trial_type, 'Condition': condition, 't': t, 'p': p})
order_res_df = pd.DataFrame(order_results)
order_res_df['p_adj'] = multipletests(order_res_df['p'], method='fdr_bh')[1]
print(order_res_df)

# Compare against chance level
ab_ratio = 0.65 / (0.65 + 0.35)  # Reward ratio for AB trials
cd_ratio = 0.75 / (0.75 + 0.25)  # Reward ratio for CD trials
ca_ratio = 0.75 / (0.75 + 0.65)  # Reward ratio for CA trials
cb_ratio = 0.75 / (0.75 + 0.35)  # Reward ratio for CB trials
ad_ratio = 0.65 / (0.65 + 0.25)  # Reward ratio for AD trials
bd_ratio = 0.35 / (0.35 + 0.25)  # Reward ratio for BD trials
random_chance = 0.5

results = []
for (trial_type, condition), subdf in E2_summary.groupby(["TrialType", "Condition"]):
    t, p = ttest_1samp(subdf["BestOption"], cb_ratio)
    results.append({"TrialType": trial_type, "Condition": condition, "t": t, "p": p, "mean": subdf["BestOption"].mean()})

res_df = pd.DataFrame(results)
print(ttest_1samp(subdf["BestOption"], 0.5))

res_df["p_adj"] = multipletests(res_df["p"], method="fdr_bh")[1]

# Paired-sample t-test for CA trials
t_stat, p_value = stats.ttest_rel(E2_summary_CA[E2_summary_CA['Condition'] == 'Baseline']['BestOption'],
                                  E2_summary_CA[E2_summary_CA['Condition'] == 'Frequency']['BestOption'])
print(f"Paired-sample t-test for CA trials: t-statistic = {t_stat}, p-value = {p_value}")

# Compare against reward ratio
reward_ratio = (0.75) / (0.65 + 0.75)  # Reward ratio for CA trials
t_baseline_rr, p_baseline_rr = ttest_1samp(E2_summary_CA[E2_summary_CA['Condition'] == 'Baseline']['BestOption'], reward_ratio)
t_frequency_rr, p_frequency_rr = ttest_1samp(E2_summary_CA[E2_summary_CA['Condition'] == 'Frequency']['BestOption'], reward_ratio)
print(f"Baseline Condition vs Reward Ratio: t-statistic = {t_baseline_rr}, p-value = {p_baseline_rr}")
print(f"Frequency Condition vs Reward Ratio: t-statistic = {t_frequency_rr}, p-value = {p_frequency_rr}")
print(f'Baseline Condition Mean: {E2_summary_CA[E2_summary_CA['Condition'] == 'Baseline']['BestOption'].mean()}; '
      f'standard deviation: {E2_summary_CA[E2_summary_CA['Condition'] == 'Baseline']['BestOption'].std()}')
print(f'Frequency Condition Mean: {E2_summary_CA[E2_summary_CA['Condition'] == 'Frequency']['BestOption'].mean()}; '
        f'standard deviation: {E2_summary_CA[E2_summary_CA['Condition'] == 'Frequency']['BestOption'].std()}')


# # Mixed ANOVA
# mixed_anova_results = pg.mixed_anova(E2_summary_CA, dv='BestOption', within='Condition', between='group_training',
#                                      subject='Subnum')
# # Print means of the BestOption for each condition
# means = E2_summary_CA.groupby(['Condition', 'group_baseline'])['BestOption'].mean().reset_index()
# print("Means of BestOption for each condition:")
# print(means)
#
# # C_diff analysis
# c_diff = E2_summary_CA.groupby('Subnum').agg(
#     C_diff=('C_diff', 'mean'),
#     group_baseline=('group_baseline', 'first'),
#     group_frequency=('group_frequency', 'first'),
#     group_training=('group_training', 'first'),
# ).reset_index()
#
# mixed_anova_results_c = pg.anova(c_diff, dv='C_diff', between='group_baseline')


# # ======================================================================================================================
# # E1 (Unused)
# # ======================================================================================================================
# # Read in the E1_summary
# E1_data = pd.read_csv('./data/E1_data_with_assignments.csv')
# E1_summary = pd.read_csv('./data/E1_summary_with_assignments.csv')
# E1_data['Phase'] = (E1_data.groupby(['Subnum', 'Condition']).cumcount() // 20) + 1
# print(E1_data.columns)
#
# # Rename the trial type
# E1_data.rename(columns={'SetSeen.': 'TrialType'}, inplace=True)
# E1_data['TrialType'] = E1_data['TrialType'].replace({0: 'AB', 1: 'CD', 2: 'CA', 3: 'CB', 4: 'AD', 5: 'BD'})
# E1_summary.rename(columns={'SetSeen.': 'TrialType'}, inplace=True)
# E1_summary['TrialType'] = E1_summary['TrialType'].replace({0: 'AB', 1: 'CD', 2: 'CA', 3: 'CB', 4: 'AD', 5: 'BD'})
#
# # Subset the data for training trials
# E1_training = E1_data[E1_data['TrialType'].isin(['AB', 'CD'])].copy()
# E1_training_summary = E1_summary[E1_summary['TrialType'].isin(['AB', 'CD'])].copy()
#
# # calculate overall training accuracy
# print(E1_summary.groupby(['Condition', 'TrialType'])['BestOption'].describe())
# weight_map = {
#     ('GainsEF', 'AB'):       0.5,
#     ('GainsEF', 'CD'):       0.5,
#     ('Gains', 'AB'):         2/3,
#     ('Gains', 'CD'):         1/3,
# }
# E1_training_summary['weight'] = E1_training_summary.apply(lambda row: weight_map[(row.Condition, row.TrialType)], axis=1)
# E1_training_summary['weighted_acc'] = E1_training_summary['BestOption'] * E1_training_summary['weight']
# training_acc = (E1_training_summary.groupby(['Subnum', 'Condition'])['weighted_acc'].sum().reset_index(name='training_accuracy'))
# E1_training_summary = E1_training_summary.merge(training_acc, on=['Subnum', 'Condition'], how='left')
# E1_training_summary.drop(columns=['weight', 'weighted_acc'], inplace=True)
#
# # median split for training accuracy
# E1_training_summary['group_training'] = (E1_training_summary.groupby('Condition')['training_accuracy']
#                                          .transform(lambda x: np.where(x >= x.quantile(0.75), 'High',
#                                                                        np.where(x <= x.quantile(0.25), 'Low', 'Average'))))
# E1_group_training = E1_training_summary[['Subnum', 'Condition', 'training_accuracy', 'group_training']].drop_duplicates()
#
# E1_added_info = E1_training_summary[['Subnum', 'Condition', 'training_accuracy', 'group_training']].drop_duplicates()
#
# # Merge the group training information into E1_summary
# E1_data = E1_data.merge(E1_added_info, on=['Subnum', 'Condition'], how='left')
# E1_summary = E1_summary.merge(E1_added_info, on=['Subnum', 'Condition'], how='left')
#
# E1_data.to_csv('./data/E1_data_full.csv', index=False)
# E1_summary.to_csv('./data/E1_summary_full.csv', index=False)