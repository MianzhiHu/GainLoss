import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from utility import normality_test, bimodal_test

# Read in the data
data = pd.read_csv('./data/data.csv')

# Subset the data
ABoptimal = data[data['ChoiceSet'] == 'AB']
CDoptimal = data[data['ChoiceSet'] == 'CD']
CAoptimal = data[data['ChoiceSet'] == 'CA']
BDoptimal = data[data['ChoiceSet'] == 'BD']
CBoptimal = data[data['ChoiceSet'] == 'CB']
ADoptimal = data[data['ChoiceSet'] == 'AD']

# set list of trials and conditions
trial_list = [ABoptimal, CDoptimal, CAoptimal, BDoptimal, CBoptimal, ADoptimal]
condition_list = ['Losses', 'LossesEF', 'Gains', 'GainsEF']


# # check the sample size for each condition
# print(ABoptimal['Condition'].value_counts())

# # Plot the overall distribution of PropOptimal
# sns.displot(data=data, x='PropOptimal', hue='ChoiceSet', kind='kde', cut=0, clip=(0, 1))
# plt.show()

# # Plot the distribution of PropOptimal for each trial with histogram in facets
# sns.displot(data=ABoptimal, x='PropOptimal', col='Condition', kind='hist')
# plt.show()

# # Plot a Q-Q plot for the distribution of PropOptimal for each trial
# stats.probplot(ABoptimal['PropOptimal'], plot=plt)
# plt.show()

# # Plot the distribution of PropOptimal for each trial
# fig, axes = plt.subplots(3, 2, figsize=(15, 10))
#
# # Subset the data based on ChoiceSet values
# trial_data = {
#     'AB': data[data['ChoiceSet'] == 'AB'],
#     'CD': data[data['ChoiceSet'] == 'CD'],
#     'CA': data[data['ChoiceSet'] == 'CA'],
#     'BD': data[data['ChoiceSet'] == 'BD'],
#     'CB': data[data['ChoiceSet'] == 'CB'],
#     'AD': data[data['ChoiceSet'] == 'AD']
# }
#
# # Create a 3x2 grid of plots
# fig, axes = plt.subplots(3, 2, figsize=(10, 18))
#
# # Loop through each subset of data and plot
# for i, (name, trial) in enumerate(trial_data.items()):
#     sns.kdeplot(data=trial, x='PropOptimal', hue='Condition', ax=axes[i // 2, i % 2])
#     axes[i // 2, i % 2].set_title(name)
#     axes[i // 2, i % 2].set_xlim(0, 1)  # Ensuring all plots have the same x-axis limit
#
# # Adjust the layout for a neat look
# plt.tight_layout()
# plt.show()

# first, test the normality of the distribution for each trial overall
# normal distribution failed (except for AB at the 1% level)
# logistic distribution failed (except for AB at the 1% level)
# gumbel distribution failed (except for AB at the 1% level)
normality_overall_shapiro = normality_test(trial_list, 'shapiro')
normality_overall_anderson = normality_test(trial_list, 'anderson')

# second, test the normality of the distribution for each trial by condition
normality_by_condition_shapiro = normality_test(trial_list, 'shapiro', condition_list)
normality_by_condition_anderson = normality_test(trial_list, 'anderson', condition_list)

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

# # Fitting a Gaussian Mixture Model with 2 components (bimodal) to the data
# # we need user-defined fit function to initialize the means
bimodal_results = bimodal_test(trial_list, condition_list)
bimodal_results_overall = bimodal_test(trial_list, None)


