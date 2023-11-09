import pandas as pd
import numpy as np
import scipy.stats as stats
from utilities.utility_tests import correlation_test
import matplotlib.pyplot as plt

# Read in the data
data = pd.read_csv('./data/data.csv')
trimodal_CA = pd.read_csv('./data/trimodal_assignments_CA.csv')

# add a index column to trimodal_CA
trimodal_CA['Subnum'] = trimodal_CA.index + 1

# merge the data with trimodal_CA
data = pd.merge(data, trimodal_CA, on='Subnum')
# data.to_csv('./data/data_with_assignment.csv', index=False)

# # explore the basic rate of optimal choices
# # visualize as four conditions and six trials
# subset = data[data['ChoiceSet'] == "CA"]
# aggregated = subset.groupby('Condition')['PropOptimal'].mean().reset_index()
#
# plt.bar(aggregated['Condition'], aggregated['PropOptimal'])
# plt.ylabel('Proportion of Optimal Choices')
# plt.ylim(0, 1)
# plt.show()


# # test all the correlations
#
# # trials with significant correlations (PropOptimal):
# # AB: Losses, LossesEF, Gains
# # CD: Losses, Gains
# # CA: LossesEF
# # CB: Losses, Gains
# # AD: Losses, LossesEF
# # BD: LossesEF, GainsEF
# # trials with significant correlations (RT):
# # AB: Losses
# # CD: LossesEF, Gains
# # CA: Losses, LossesEF
# # CB: Losses, LossesEF
# # AD: Losses, LossesEF
# # BD: Losses, LossesEF
#
trial_list = ['AB', 'CD', 'CA', 'CB', 'AD', 'BD']
condition_list = ['Losses', 'LossesEF', 'Gains', 'GainsEF']
loss_list = ['Losses', 'LossesEF']
variable_of_interest_list = ['RT']
#
# for trial in trial_list:
#     for condition in condition_list:
#         for variable_of_interest in variable_of_interest_list:
#             correlation_test(data, trial, variable_of_interest, condition, checker=True)
#
#
# correlation_test(data, 'BD', 'PropOptimal', 'GainsEF', sig_only=True)

# so, we are back to individual differences
# we need more data to increase the power
probability_list = ['prob1', 'prob2', 'prob3']

for probability in probability_list:
    for condition in condition_list:
        correlation_test(data, 'CA', probability, condition, checker=True)

for probability in probability_list:
    correlation_test(data, 'CA', probability, checker=True)

CA_prob1_all = correlation_test(data, 'CA', 'prob1', sig_only=True)
CA_prob1 = correlation_test(data, 'CA', 'prob1', condition='Gains', sig_only=True)

# do some t-tests
group1 = data[data['assignments'] == 1]
group2 = data[data['assignments'] == 2]
group3 = data[data['assignments'] == 3]

for scale in data.columns[4:20]:
    print(scale)
    print(stats.f_oneway(group1[scale], group2[scale], group3[scale]))
    print('')

for scale in data.columns[4:20]:
    print(scale)
    print(stats.ttest_ind(group3[scale], group1[scale], equal_var=False))
    print('')


# Variables to plot
variables = ['ESIBF_SubstanceUse', 'Big5E', 'TPM_Boldness']
labels = ['Advantageous Learners', 'Average Learners', 'Disadvantageous Learners']

# Calculate means and standard errors
means = [[np.mean(group[var]) for group in [group1, group2, group3]] for var in variables]
std_errors = [[np.std(group[var]) / np.sqrt(len(group[var])) for group in [group1, group2, group3]] for var in variables]

# Number of groups and number of variables
n_groups = len(labels)
n_vars = len(variables)

# The x location for the groups
ind = np.arange(n_groups)
# The width of the bars
width = 0.8 / n_vars
# Colors for each variable
colors = ['#1F77B4', '#FF7F0E', '#2CA02C']

# Create the plot
# Use a clear and modern style for the plot
plt.style.use('seaborn-white')
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars for each variable
for i in range(n_vars):
    # Offset each bar by the width of the bars so they don't overlap
    bar_positions = ind + i * width
    ax.bar(bar_positions, means[i], width, yerr=std_errors[i], capsize=5, color=colors[i], label=variables[i], alpha=0.5)

# Add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Group-Level Individual Differences', fontsize=24)
ax.set_xticks(ind + width / n_vars + 0.2)
ax.set_xticklabels(labels, ha='center')

# Add a legend
ax.legend(title='Variables', loc='upper left')

# Remove background and grid
plt.gca().set_facecolor('none')
plt.grid(False)

# Adjust spines to be less prominent
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_edgecolor('gray')
plt.gca().spines['bottom'].set_edgecolor('gray')

# Show the plot
plt.tight_layout()
plt.show(dpi=600)
