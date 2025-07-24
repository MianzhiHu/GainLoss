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

# Extract the difference of C choice rate across conditions
E2_summary_CA = E2_summary[E2_summary['SetSeen '] == 2]
c_diff = (E2_summary_CA[E2_summary_CA['Condition'] == 'Baseline']['BestOption'].values -
          E2_summary_CA[E2_summary_CA['Condition'] == 'Frequency']['BestOption'].values)
E2_summary_CA['C_diff'] = np.repeat(c_diff, 2)

# Mixed ANOVA
mixed_anova_results = pg.mixed_anova(E2_summary_CA, dv='BestOption', within='Condition', between='group_baseline',
                                     subject='Subnum')
# Print means of the BestOption for each condition
means = E2_summary_CA.groupby(['Condition', 'group_baseline'])['BestOption'].mean().reset_index()
print("Means of BestOption for each condition:")
print(means)

# plot the mixed ANOVA results
plt.figure(figsize=(10, 6))
sns.barplot(x='Condition', y='BestOption', hue='group_baseline', data=E2_summary_CA, errorbar=('ci', 95))
plt.title('Mixed ANOVA Results: BestOption by Condition and Group')
plt.ylabel('Best Option (Proportion of C Choices)')
plt.xlabel('Condition')
plt.axhline(0.5, color='darkred', linestyle='--')  # Add a horizontal line at 0.5
sns.despine()
plt.savefig('./figures/mixed_anova_results.png', dpi=600)
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Training performance analysis
# ----------------------------------------------------------------------------------------------------------------------
E2_training = E2_data[E2_data['SetSeen '].isin([0, 1])].copy()
print(E2_training['SetSeen '].value_counts())
print(E2_training.groupby(['Condition', 'group_baseline'])['KeyResponse'].value_counts(normalize=True).unstack().fillna(0))

# Plot selection rate per phase
plt.figure(figsize=(10, 6))
sns.lineplot(x='Phase', y='BestOption', hue='group_baseline', data=E2_training[E2_training['Condition'] == 'Frequency'])
plt.ylabel('Best Option Selection Rate')
plt.xlabel('Phase')
plt.title('Best Option Selection by Phase')
plt.savefig('./figures/phase_learning.png', dpi=600)
plt.show()

mixed_anova_training = pg.mixed_anova(E2_summary[E2_summary['SetSeen '].isin([0, 1])], dv='BestOption',
                                      within='Condition', between='group_baseline', subject='Subnum')
plt.figure(figsize=(10, 6))
sns.barplot(x='Condition', y='BestOption', hue='group_baseline', data=E2_summary[E2_summary['SetSeen '].isin([1])], errorbar=('ci', 95))
plt.title('Mixed ANOVA Results: BestOption by Condition and Group')
plt.ylabel('Best Option (Proportion of C Choices)')
plt.xlabel('Condition')
plt.axhline(0.5, color='darkred', linestyle='--')  # Add a horizontal line at 0.5
sns.despine()
plt.savefig('./figures/mixed_anova_training_results_CD.png', dpi=600)
plt.show()



# change the condition to a categorical variable
E2_summary.rename(columns={'SetSeen ': 'TrialType'}, inplace=True)

# Rename the trial type
E2_summary['TrialType'] = E2_summary['TrialType'].replace({0: 'AB', 1: 'CD', 2: 'CA', 3: 'CB', 4: 'AD', 5: 'BD'})

# plot the E2_summary
sns.set(style='white')
plt.figure(figsize=(10, 6))
sns.barplot(x='TrialType', y='BestOption', hue='Condition', data=E2_summary, errorbar=('ci', 95))
plt.title('Proportion of Optimal Choices')
plt.ylabel('Proportion of Optimal Choices')
plt.xlabel('Trial Type')
# add a horizontal line at 0.5
plt.axhline(0.5, color='darkred', linestyle='--')
sns.despine()
plt.savefig('./figures/propoptimal_all.png', dpi=600)
plt.show()

