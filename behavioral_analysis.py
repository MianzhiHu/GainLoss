import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Read in the E2_summary
E2_data = pd.read_csv('./data/E2_data_with_assignments.csv')
E2_summary = pd.read_csv('./data/E2_summary_with_assignments.csv')
E2_data['Phase'] = (E2_data.groupby(['Subnum', 'Condition']).cumcount() // 20) + 1
print(E2_data.columns)

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

E2_summary.to_csv('./data/E2_summary_full.csv', index=False)
E2_summary_testing.to_csv('./data/E2_summary_testing.csv', index=False)
E2_summary_CA.to_csv('./data/E2_summary_CA.csv', index=False)
E2_data.to_csv('./data/E2_data_full.csv', index=False)
E2_data_testing.to_csv('./data/E2_data_testing.csv', index=False)

# Correlation
# wider the df
training_acc_wider = training_acc.pivot(index='Subnum', columns='Condition', values='training_accuracy').reset_index()
E2_CA_wider = E2_summary_CA.pivot(index='Subnum', columns='Condition', values='BestOption').reset_index()
corr = pg.corr(E2_CA_wider['Baseline'],
               E2_CA_wider['Frequency'], method='spearman')

# Mixed ANOVA
mixed_anova_results = pg.mixed_anova(E2_summary_CA, dv='BestOption', within='Condition', between='group_training',
                                     subject='Subnum')
# Print means of the BestOption for each condition
means = E2_summary_CA.groupby(['Condition', 'group_baseline'])['BestOption'].mean().reset_index()
print("Means of BestOption for each condition:")
print(means)

# C_diff analysis
c_diff = E2_summary_CA.groupby('Subnum').agg(
    C_diff=('C_diff', 'mean'),
    group_baseline=('group_baseline', 'first'),
    group_frequency=('group_frequency', 'first'),
    group_training=('group_training', 'first'),
).reset_index()

mixed_anova_results_c = pg.anova(c_diff, dv='C_diff', between='group_baseline')
# Print the c_diff mixed ANOVA results



# chi-square test of independence for group_baseline and group_training
p_grouping = E2_summary_CA.groupby('Subnum')[['group_baseline', 'group_frequency', 'group_training']].first().reset_index()
p_grouping['group_baseline'] = p_grouping['group_baseline'].replace({1: 'baseline_Advantageous',
                                                                    2: 'baseline_Average',
                                                                    3: 'baseline_Disadvantageous'})
p_grouping['group_frequency'] = p_grouping['group_frequency'].replace({1: 'frequency_Advantageous',
                                                                        2: 'frequency_Average',
                                                                        3: 'frequency_Disadvantageous'})
contingency_table = pd.crosstab(p_grouping['group_baseline'], p_grouping['group_frequency'])
#
# from itertools import combinations
# from statsmodels.stats.multitest import multipletests
# from scipy.stats import chi2_contingency
#
#
# def chisq_and_posthoc_corrected(df, p_adjust="fdr_bh"):
#     """Receives a dataframe and performs chi2 test and then post hoc.
#     Prints the p-values and corrected p-values (after FDR correction)"""
#     # start by running chi2 test on the matrix
#     chi2, p, dof, ex = chi2_contingency(df, correction=True)
#     print(f"Chi2 result of the contingency table: {chi2}, p-value: {p}, dof: {dof}")
#
#     # now run post hoc test
#     col_totals = df.sum(axis=0)
#     n_cols = len(col_totals)
#     row_totals = df.sum(axis=1)
#     n_rows = len(row_totals)
#     n_total = sum(col_totals)
#
#     adj_res_results = []
#     for i in range(n_rows):
#         for j in range(n_cols):
#             adj_res = (df.iloc[i, j] - ex[i, j]) / (ex[i, j] * (1 - row_totals[i] / n_total) * (1 - col_totals[j] / n_total)) ** 0.5
#             adj_res_results.append((df.index[i], df.columns[j], adj_res))
#     adj_res_df = pd.DataFrame(adj_res_results, columns=['Row', 'Column', 'Adj_Residual'])
#     adj_res_df['p_value'] = 2 * (1 - stats.norm.cdf(np.abs(adj_res_df['Adj_Residual'])))
#     adj_res_df['p_value_corrected'] = multipletests(adj_res_df['p_value'], method=p_adjust)[1]
#     print("Post hoc results with FDR correction:")
#     print(adj_res_df)
#
#     return chi2, adj_res_df
#
#
#
# # Perform chi-square test and post-hoc analysis
# chisq, residual_df = chisq_and_posthoc_corrected(contingency_table)
# residual_table = residual_df.pivot(
#     index="Row",
#     columns="Column",
#     values="Adj_Residual"
# )
#
# # plot the mixed ANOVA results
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Condition', y='BestOption', hue='group_training', data=E2_summary_CA, errorbar=('ci', 95))
# plt.title('Mixed ANOVA Results: BestOption by Condition and Group')
# plt.ylabel('Best Option (Proportion of C Choices)')
# plt.xlabel('Condition')
# plt.axhline(0.5, color='darkred', linestyle='--')  # Add a horizontal line at 0.5
# sns.despine()
# plt.savefig('./figures/mixed_anova_results.png', dpi=600)
# plt.show()
#
# # plot linear plot
# plt.figure(figsize=(10, 6))
# sns.lmplot(data=E2_summary_CA, x='training_accuracy', y='BestOption', hue='Condition', height=6, aspect=1.2,
#            markers=['o','s'], scatter_kws={'alpha':0.4, 's':50}, ci=95)
# plt.xlabel('Training Accuracy')
# plt.ylabel('Best Option (Proportion of C Choices)')
# plt.title('Best Option vs. Training Accuracy by Condition')
# plt.tight_layout()
# plt.savefig('./figures/best_option_vs_training_accuracy.png', dpi=600)
# plt.show()
#
# # plot the number of participants in each group_baseline category per order
# plt.figure(figsize=(10, 6))
# sns.countplot(x='group_frequency', hue='order', data=E2_summary_CA[['Subnum', 'group_baseline', 'group_frequency', 'order']].drop_duplicates(),
#               palette='Set2')
# plt.title('Number of Participants in Each Group Baseline Category')
# plt.xlabel('Group Baseline')
# plt.ylabel('Number of Participants')
# plt.xticks(labels=['Advantageous', 'Average', 'Disadvantageous'], ticks=np.arange(3))
# plt.savefig('./figures/group_baseline_counts.png', dpi=600)
# plt.show()
#
# # ----------------------------------------------------------------------------------------------------------------------
# # Training performance analysis
# # ----------------------------------------------------------------------------------------------------------------------
# print(E2_training.groupby(['Condition', 'group_baseline'])['KeyResponse'].value_counts(normalize=True).unstack().fillna(0))
#
# # Plot selection rate per phase
# plt.figure(figsize=(10, 6))
# sns.lineplot(x='Phase', y='BestOption', hue='group_baseline', data=E2_training[E2_training['Condition'] == 'Frequency'])
# plt.ylabel('Best Option Selection Rate')
# plt.xlabel('Phase')
# plt.title('Best Option Selection by Phase')
# plt.savefig('./figures/phase_learning.png', dpi=600)
# plt.show()
#
# mixed_anova_training = pg.mixed_anova(E2_summary[E2_summary['TrialType'].isin(['AB', 'CD'])], dv='BestOption',
#                                       within='Condition', between='group_baseline', subject='Subnum')
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Condition', y='BestOption', hue='group_baseline', data=E2_summary[E2_summary['TrialType'].isin(['CD'])], errorbar=('ci', 95))
# plt.title('Mixed ANOVA Results: BestOption by Condition and Group')
# plt.ylabel('Best Option (Proportion of C Choices)')
# plt.xlabel('Condition')
# plt.axhline(0.5, color='darkred', linestyle='--')  # Add a horizontal line at 0.5
# sns.despine()
# plt.savefig('./figures/mixed_anova_training_results_CD.png', dpi=600)
# plt.show()
#
# # plot the E2_summary
# sns.set(style='white')
# plt.figure(figsize=(10, 6))
# sns.barplot(x='TrialType', y='BestOption', hue='Condition', data=E2_summary, errorbar=('ci', 95))
# plt.title('Proportion of Optimal Choices')
# plt.ylabel('Proportion of Optimal Choices')
# plt.xlabel('Trial Type')
# # add a horizontal line at 0.5
# plt.axhline(0.5, color='darkred', linestyle='--')
# sns.despine()
# plt.savefig('./figures/propoptimal_all.png', dpi=600)
# plt.show()
#
# # Plot Choice Distribution per Choice Type and Condition using Facet Grid
# g = sns.FacetGrid(E2_summary, col='Condition', hue='TrialType', height=4, aspect=1.5)
# g.map_dataframe(sns.kdeplot, x='BestOption', clip=(0, 1))
# g.set_axis_labels('Best Option', 'Density')
# g.set_titles(col_template='{col_name}')
# g.add_legend()
# plt.subplots_adjust(top=0.9)
# g.fig.suptitle('Choice Distribution by Trial Type for each Condition')
# plt.tight_layout()
# plt.savefig('./figures/choice_distribution_per_choice_type_and_condition.png', dpi=600)
# plt.show()
#
# g = sns.FacetGrid(E2_training_summary, col='Condition', height=4, aspect=1.5)
# g.map_dataframe(sns.kdeplot, x='BestOption', clip=(0, 1))
# g.set_axis_labels('Best Option', 'Density')
# g.set_titles(col_template='{col_name}')
# g.add_legend()
# plt.subplots_adjust(top=0.9)
# g.fig.suptitle('Choice Distribution by Trial Type for each Condition')
# plt.tight_layout()
# plt.savefig('./figures/choice_distribution_per_choice_type_and_condition.png', dpi=600)
# plt.show()


