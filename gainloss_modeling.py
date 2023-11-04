import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities.utility_ComputationalModeling import (ComputationalModels, likelihood_ratio_test, dict_generator,
                                                     best_param_generator)
from scipy.stats import pearsonr, spearmanr, ttest_ind
from utilities.utility_distribution import best_fitting_participants

# Read in the data
data = pd.read_csv('./data/ABCDGainsLossesData_F2023.csv')
assignment = pd.read_csv('./data/trimodal_assignments_CA.csv')

# result_good = pd.read_csv('./data/decayfre_good_learners.csv')
# result_bad = pd.read_csv('./data/decayfre_bad_learners.csv')

# data_CA = data[data['SetSeen.'] == 2]
# for name, group in data_CA.groupby('Subnum'):
#     if group['ReactTime'].mean() > 10000:
#         print(name)

# remove the outlier trials (RT > 10s)
# other files don't have this problem
data = data[data['Subnum'] != 122]

# reindex the subnum
data = data.reset_index(drop=True)
data.iloc[:, 1] = (data.index // 250) + 1
data['KeyResponse'] = data['KeyResponse'] - 1

# copy the index of all participants assigned to 3
bad_learners = assignment[assignment['assignments'] == 3].index.tolist()
bad_learners = [x + 1 for x in bad_learners]

good_learners = assignment[assignment['assignments'] == 1].index.tolist()
good_learners = [x + 1 for x in good_learners]

# sample one participant to serve as an example
participant = data[data['Subnum'] == 6]

# keep only the participants assigned to bad learners
bad_learners_data = data[data['Subnum'].isin(bad_learners)]
bad_learners_data = bad_learners_data[bad_learners_data['SetSeen.'].isin([0, 1, 2])]
bad_learners_data = bad_learners_data[bad_learners_data['Condition'] == 'Gains']

# keep only the participants assigned to good learners
good_learners_data = data[data['Subnum'].isin(good_learners)]
good_learners_data = good_learners_data[good_learners_data['SetSeen.'].isin([0, 1, 2])]
good_learners_data = good_learners_data[good_learners_data['Condition'] == 'Gains']
print(good_learners_data['SetSeen.'].unique())

# convert into dictionary
data_dict = dict_generator(data)
bad_learners_dict = dict_generator(bad_learners_data)
good_learners_dict = dict_generator(good_learners_data)
participant_dict = dict_generator(participant)

# set up the reward structure
reward_means = [0.65, 0.35, 0.75, 0.25]
reward_sd = [0.43, 0.43, 0.43, 0.43]

# fit the data
model_decayfre = ComputationalModels(reward_means, reward_sd,
                                     model_type='decay_fre', condition='Both', num_trials=250)

model_decay = ComputationalModels(reward_means, reward_sd,
                                  model_type='decay', condition='Both', num_trials=250)

model_sampler_decay = ComputationalModels(reward_means, reward_sd,
                                          model_type='sampler_decay', condition='Both', num_trials=250, num_params=2)


# # use a sample to test whether the model is functioning
# sample = model_sampler_decay.fit(participant_dict, num_iterations=100)
# sample_df = pd.DataFrame(sample)

# fit the model with all participants
results_data_decay = model_decay.fit(data_dict, num_iterations=1000)
results_data_decay = pd.DataFrame(results_data_decay)
results_data_decay.iloc[:, 3] = results_data_decay.iloc[:, 3].astype(str)
# results_data_decay.to_csv('./data/decayfre_data.csv', index=False)

print(results_data_decay['AIC'].mean())
print(results_data_decay['BIC'].mean())

# results_good = model_sampler_decay.fit(good_learners_dict, num_iterations=1000)
#
# result_good = pd.DataFrame(results_good)
# result_good.iloc[:, 3] = result_good.iloc[:, 3].astype(str)
# # result_good.to_csv('./data/decay_good_learners.csv', index=False)
#
# # sum up the AIC column
# print(result_good['AIC'].mean())
# print(result_good['BIC'].mean())

# results_bad = model_sampler_decay.fit(bad_learners_dict, num_iterations=1000)
#
# result_bad = pd.DataFrame(results_bad)
# result_bad.iloc[:, 3] = result_bad.iloc[:, 3].astype(str)
# # result_bad.to_csv('./data/decay_bad_learners.csv', index=False)
#
# # sum up the AIC column
# print(result_bad['AIC'].mean())
# print(result_bad['BIC'].mean())
#
# # extract the best beta
# best_t_good = best_param_generator(result_good, 't')
# best_alpha_good = best_param_generator(result_good, 'a')
# best_beta_good = best_param_generator(result_good, 'b')
# best_t_bad = best_param_generator(result_bad, 't')
# best_alpha_bad = best_param_generator(result_bad, 'a')
# best_beta_bad = best_param_generator(result_bad, 'b')
#
# # print out the mean and sd of the best beta
# print(sum(best_beta_good) / len(best_beta_good))
# print(sum(best_beta_bad) / len(best_beta_bad))
#
# # print out the mean and sd of the best t
# print(sum(best_t_good) / len(best_t_good))
# print(sum(best_t_bad) / len(best_t_bad))
#
# # print out the mean and sd of the best alpha
# print(sum(best_alpha_good) / len(best_alpha_good))
# print(sum(best_alpha_bad) / len(best_alpha_bad))
#
# # conduct t-test
# ttest_ind(best_t_good, best_t_bad)
# ttest_ind(best_beta_good, best_beta_bad)
# ttest_ind(best_alpha_good, best_alpha_bad)
#
# pearsonr(best_t_bad, best_beta_bad)
# pearsonr(best_t_good, best_beta_good)

