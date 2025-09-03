import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

# Read in the E2_summary
data = pd.read_csv('data.csv')
summary = pd.read_csv('data_summary.csv')
summary_CA = summary[summary['TrialType'] == 'CA'].reset_index(drop=True)

# ======================================================================================================================
# Data Analysis
# ======================================================================================================================
# Demographics
print(data.groupby('Subnum')['Sex'].first().value_counts()) # Gender
print(data.groupby('Subnum')['Age'].first().describe()) # Age
print(data.groupby('Subnum')['Race'].first().value_counts()) # Race
print(data.groupby('Subnum')['Ethnicity'].first().value_counts()) # Ethnicity
print(data.groupby('Subnum')['order'].first().value_counts()) # Task Completion Order

# Paired-sample t-test for CA trials (Figure 2c)
t_stat, p_value = stats.ttest_rel(summary_CA[summary_CA['Condition'] == 'Baseline']['BestOption'],
                                  summary_CA[summary_CA['Condition'] == 'Frequency']['BestOption'])
print(f"Paired-sample t-test for CA trials: t-statistic = {t_stat}, p-value = {p_value}")

# Compare C choice rates in CA trials against the reward ratio (Figure 2c)
reward_ratio = (0.75) / (0.65 + 0.75)  # Reward ratio for CA trials
t_baseline_rr, p_baseline_rr = ttest_1samp(summary_CA[summary_CA['Condition'] == 'Baseline']['BestOption'], reward_ratio)
t_frequency_rr, p_frequency_rr = ttest_1samp(summary_CA[summary_CA['Condition'] == 'Frequency']['BestOption'], reward_ratio)
print(f"Baseline Condition vs Reward Ratio: t-statistic = {t_baseline_rr}, p-value = {p_baseline_rr}")
print(f"Frequency Condition vs Reward Ratio: t-statistic = {t_frequency_rr}, p-value = {p_frequency_rr}")
print(f'Baseline Condition Mean: {summary_CA[summary_CA['Condition'] == 'Baseline']['BestOption'].mean()}; '
      f'standard deviation: {summary_CA[summary_CA['Condition'] == 'Baseline']['BestOption'].std()}')
print(f'Frequency Condition Mean: {summary_CA[summary_CA['Condition'] == 'Frequency']['BestOption'].mean()}; '
        f'standard deviation: {summary_CA[summary_CA['Condition'] == 'Frequency']['BestOption'].std()}')

# Order effects (Supplementary Table 1)
order_data = data.groupby(['Subnum', 'Condition', 'TrialType', 'order'])['BestOption'].mean().reset_index()

# for each trial type in each condition, compare the two orders
order_results = []
for (trial_type, condition), subdf in order_data.groupby(['TrialType', 'Condition']):
    t, p = stats.ttest_ind(subdf[subdf['order'] == 'BF']['BestOption'],
                           subdf[subdf['order'] == 'FB']['BestOption'])
    order_results.append({'TrialType': trial_type, 'Condition': condition, 't': t, 'p': p})
order_res_df = pd.DataFrame(order_results)
order_res_df['p_adj'] = multipletests(order_res_df['p'], method='fdr_bh')[1]
print(order_res_df)

# One-sample t-tests against chance level (0.5) for each TrialType and Condition (Supplementary Table 3)
results = []
for (trial_type, condition), subdf in summary.groupby(["TrialType", "Condition"]):
    t, p = ttest_1samp(subdf["BestOption"], 0.500)
    results.append({"TrialType": trial_type, "Condition": condition, "t": t, "p": p, "mean": subdf["BestOption"].mean()})

res_df = pd.DataFrame(results)
res_df["p_adj"] = multipletests(res_df["p"], method="fdr_bh")[1]