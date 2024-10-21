import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from utilities.utility_tests import normality_test, bimodal_test


# Read in the data
data = pd.read_csv('./data/data_PropOptimal.csv')
data_with_assignment = pd.read_csv('./data/data_with_assignment.csv')

print(data_with_assignment['Ethnicity'].value_counts()/6)

# Subset the data
ABoptimal = data[data['ChoiceSet'] == 'AB']
CDoptimal = data[data['ChoiceSet'] == 'CD']
CAoptimal = data[data['ChoiceSet'] == 'CA']
BDoptimal = data[data['ChoiceSet'] == 'BD']
CBoptimal = data[data['ChoiceSet'] == 'CB']
ADoptimal = data[data['ChoiceSet'] == 'AD']

CAoptimal_with_assignment = data_with_assignment[data_with_assignment['ChoiceSet'] == 'CA']
CA_group1 = CAoptimal_with_assignment[CAoptimal_with_assignment['assignments'] == 1]
CA_group2 = CAoptimal_with_assignment[CAoptimal_with_assignment['assignments'] == 2]
CA_group3 = CAoptimal_with_assignment[CAoptimal_with_assignment['assignments'] == 3]

CA_groups = [CA_group1, CA_group2, CA_group3]

# t-test for the difference between EF and non-EF conditions
stats.ttest_ind(CA_group2[CA_group2['Condition'] == 'Gains']['PropOptimal'],
                CA_group2[CA_group2['Condition'] == 'GainsEF']['PropOptimal'])


# set list of trials and conditions
trial_list = [ABoptimal, CDoptimal, CAoptimal, BDoptimal, CBoptimal, ADoptimal]
condition_list = ['Losses', 'LossesEF', 'Gains', 'GainsEF']

# # check the sample size for each condition
# print(ABoptimal['Condition'].value_counts())

# # Plot the overall distribution of PropOptimal
# sns.displot(data=data, x='RT', hue='ChoiceSet', kind='kde')
# plt.show()

# # Plot the distribution of PropOptimal for each trial with histogram in facets
# sns.displot(data=CA_group1, x='PropOptimal', col='Condition', kind='kde')
# plt.gca().set_xlim([0, 1])
# plt.show()

# # Plot a Q-Q plot for the distribution of PropOptimal for each trial
# stats.probplot(ABoptimal['PropOptimal'], plot=plt)
# plt.show()

# # Plot the distribution of PropOptimal for each trial
# fig, axes = plt.subplots(3, 2, figsize=(15, 10))
#
# Subset the data based on ChoiceSet values
trial_data = {
    'AB': data[data['ChoiceSet'] == 'AB'],
    'CD': data[data['ChoiceSet'] == 'CD'],
    'CA': data[data['ChoiceSet'] == 'CA'],
    'BD': data[data['ChoiceSet'] == 'BD'],
    'CB': data[data['ChoiceSet'] == 'CB'],
    'AD': data[data['ChoiceSet'] == 'AD']
}

group_data = {
    'Good Learners': CA_group1,
    'Average Learners': CA_group2,
    'Bad Learners': CA_group3
}

condition_order = ['Gains', 'GainsEF', 'Losses', 'LossesEF']

# # Create a 3x2 grid of plots
# fig, axes = plt.subplots(3, 2, figsize=(10, 18))
#
# # Loop through each subset of data and plot
# for i, (name, trial) in enumerate(trial_data.items()):
#     sns.kdeplot(data=trial, x='PropOptimal', hue='Condition', ax=axes[i // 2, i % 2],
#                 hue_order=condition_order)
#     axes[i // 2, i % 2].set_title(name)
#
#     # Set only the left boundary of x-axis
#     current_xlim = axes[i // 2, i % 2].get_xlim()
#     axes[i // 2, i % 2].set_xlim(0, current_xlim[1])  # 0 is the new left boundary
#
#     # axes[i // 2, i % 2].set_xlim(0.75, 1)  # Ensuring all plots have the same x-axis limit
#
# # Adjust the layout for a neat look
# plt.tight_layout()
# plt.show()

# create plots individually
ax = sns.kdeplot(data=CA_group3, x='PropOptimal', hue='Condition', hue_order=condition_order)

# Loop over each line to find the mode and draw a vertical line
lines = ax.get_lines()
for line in lines:
    x, y = line.get_data()
    mode = x[np.argmax(y)]
    color = line.get_color()  # match the color of the KDE line
    plt.axvline(mode, color=color, linestyle='--')

plt.gca().set_xlim([0, 0.2])
plt.title('Disadvantageous Learners')
plt.tight_layout()
plt.show(dpi=600)

# fit the normal distribution individually
# preallocate a list to store the results
normal_results = []

for groups in CA_groups:
    for condition in condition_list:
        result = stats.norm.fit(groups[groups['Condition'] == condition]['PropOptimal'])

        normal_results.append(
            {'condition': condition, 'group': groups['assignments'].iloc[0], 'mean': result[0], 'sd': result[1]})

normal_results = pd.DataFrame(normal_results, columns=['condition', 'group', 'mean', 'sd'])




# first, test the normality of the distribution for each trial overall
# normal distribution failed (except for AB at the 1% level)
# logistic distribution failed (except for AB at the 1% level)
# gumbel distribution failed (except for AB at the 1% level)
normality_overall_shapiro = normality_test(trial_list, 'shapiro', 'PropOptimal')
normality_overall_anderson = normality_test(trial_list, 'anderson', 'PropOptimal')

# second, test the normality of the distribution for each trial by condition
normality_by_condition_shapiro = normality_test(trial_list, 'shapiro', 'PropOptimal', condition_list)
normality_by_condition_anderson = normality_test(trial_list, 'anderson', 'PropOptimal', condition_list)

# now test the binomial distribution of PropOptimal for each trial
# preallocate a list to store the results
binomial_results = []

for trial in trial_list:
    for condition in condition_list:
        result = stats.binomtest(np.sum(trial[trial['Condition'] == condition]['PropOptimal'] > 0.5),
                                 len(trial[trial['Condition'] == condition]['PropOptimal']))
        binomial_results.append(
            {'trial': trial['ChoiceSet'].iloc[0], 'condition': condition, 'statistic': result.statistic,
             'p-value': result.pvalue})

binomial_results = pd.DataFrame(binomial_results, columns=['trial', 'condition', 'statistic', 'p-value'])

# check how many people are above 0.5
for condition in condition_list:
    print(condition)
    print(np.sum(CAoptimal[CAoptimal['Condition'] == condition]['PropOptimal'] > 0.5))

# # what about RT? Let's try split RT into two groups based on the median and test the difference
# # not really a good differentiator
# binomial_results_RT = []
#
# for trial in trial_list:
#     for condition in condition_list:
#         result = stats.binomtest(np.sum(trial[trial['Condition'] == condition]['RT'] > trial['RT'].median()),
#                                  len(trial[trial['Condition'] == condition]['RT']))
#         binomial_results_RT.append(
#             {'trial': trial['ChoiceSet'].iloc[0], 'condition': condition, 'statistic': result.statistic,
#              'p-value': result.pvalue})
#
# binomial_results_RT = pd.DataFrame(binomial_results_RT, columns=['trial', 'condition', 'statistic', 'p-value'])

# # now fit the beta distribution to PropOptimal
# # beta distribution is not the answer
# # preallocate a list to store the results
# beta_results = []
#
# for trial in trial_list:
#     for condition in condition_list:
#         # change the range of the data to (0, 1)
#         trial.loc[trial['PropOptimal'] == 1, 'PropOptimal'] = 0.99999
#         trial.loc[trial['PropOptimal'] == 0, 'PropOptimal'] = 0.00001
#
#         result = stats.beta.fit(trial[trial['Condition'] == condition]['PropOptimal'], floc=0, fscale=1)
#
#         # evaluate the fit
#         kstest = stats.kstest(trial[trial['Condition'] == condition]['PropOptimal'], 'beta', args=result)
#
#         # calculate the RMSE
#         # draw predicted values as the expected value of the beta distribution
#         ev = stats.beta.mean(result[0], result[1], loc=0, scale=1)
#         rmse = np.sqrt(np.mean((trial[trial['Condition'] == condition]['PropOptimal'] - ev) ** 2))
#
#         # calculate the R2
#         TSS = np.sum((trial[trial['Condition'] == condition]['PropOptimal'] -
#                       trial[trial['Condition'] == condition]['PropOptimal'].mean()) ** 2)
#         print(TSS)
#         RSS = np.sum((trial[trial['Condition'] == condition]['PropOptimal'] - ev) ** 2)
#         print(RSS)
#         R2 = 1 - RSS / TSS
#
#         beta_results.append(
#             {'trial': trial['ChoiceSet'].iloc[0], 'condition': condition, 'a': result[0], 'b': result[1],
#              'D': kstest.statistic, 'p-value': kstest.pvalue, 'RMSE': rmse, 'R2': R2})
#
# beta_results = pd.DataFrame(beta_results, columns=['trial', 'condition', 'a', 'b', 'D', 'p-value', 'RMSE', 'R2'])


# # Fitting a Gaussian Mixture Model with 2 components (bimodal) to the data
# # we need user-defined fit function to initialize the means
bimodal_results = bimodal_test(trial_list, condition_list)
bimodal_results_overall = bimodal_test(trial_list, None)

# now, we fit a Weibull distribution to RT
# generally good fit
# preallocate a list to store the results
weibull_results = []

for trial in trial_list:
    for condition in condition_list:
        result = stats.weibull_min.fit(trial[trial['Condition'] == condition]['RT'])

        # evaluate the fit
        kstest = stats.kstest(trial[trial['Condition'] == condition]['RT'], 'weibull_min', args=result)

        weibull_results.append(
            {'trial': trial['ChoiceSet'].iloc[0], 'condition': condition, 'shape': result[0], 'location': result[1],
             'scale': result[2], 'D': kstest.statistic, 'p-value': kstest.pvalue})

weibull_results = pd.DataFrame(weibull_results, columns=['trial', 'condition', 'shape', 'location',
                                                         'scale', 'D', 'p-value'])

# # visualize the distribution of parameters
# sns.displot(data=weibull_results, x='shape', kind='kde')
# plt.show()

# # test the normality of the distribution of parameters
# # the distribution of parameters are normally distributed,
# # so we can use ANOVA to test the difference between conditions
# params_weibull = ['shape', 'location', 'scale']
# for param in params_weibull:
#     print(param)
#     print(stats.shapiro(weibull_results[param]))

# # test the difference between conditions
# # we found that the difference between trials are significant
# # but the difference between conditions are not significant
# anova_results = []
#
# for param in params_weibull:
#     result = stats.f_oneway(weibull_results[weibull_results['trial'] == 'AB'][param],
#                             weibull_results[weibull_results['trial'] == 'CD'][param],
#                             weibull_results[weibull_results['trial'] == 'CA'][param],
#                             weibull_results[weibull_results['trial'] == 'BD'][param],
#                             weibull_results[weibull_results['trial'] == 'CB'][param],
#                             weibull_results[weibull_results['trial'] == 'AD'][param])
#
#     anova_results.append({'parameter': param, 'F': result.statistic, 'p-value': result.pvalue})
#
# anova_results = pd.DataFrame(anova_results, columns=['parameter', 'F', 'p-value'])
