import pandas as pd
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

# # do some t-tests
# group1 = data[data['assignments'] == 1]
# group2 = data[data['assignments'] == 2]
# group3 = data[data['assignments'] == 3]
#
# for scale in data.columns[4:20]:
#     print(scale)
#     print(stats.f_oneway(group1[scale], group2[scale], group3[scale]))
#     print('')
#
#
# data_to_be_plotted = [group1['Big5E'].mean(),
#                       group2['Big5E'].mean(),
#                       group3['Big5E'].mean()]
# labels = ['Group 1', 'Group 2', 'Group 3']
#
# plt.bar(labels, data_to_be_plotted)
# plt.ylabel('Mean Bis11Score')
# plt.show()

