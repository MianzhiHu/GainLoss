import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities.utility_ComputationalModeling import (ComputationalModels, likelihood_ratio_test, dict_generator,
                                                     best_param_generator, bayes_factor)
from scipy.stats import pearsonr, spearmanr, ttest_ind
from utilities.utility_distribution import best_fitting_participants

# Read in the data
data = pd.read_csv('./data/data.csv')
propOptimal = pd.read_csv('./data/data_with_assignment.csv')
propOptimal_CA = propOptimal[propOptimal['ChoiceSet'] == 'CA']
assignment = pd.read_csv('./data/trimodal_assignments_CA.csv')

# decay_good = pd.read_csv('./data/decay_good_learners.csv')
# decay_bad = pd.read_csv('./data/decay_bad_learners.csv')
# decayfre_good = pd.read_csv('./data/decayfre_good_learners.csv')
# decayfre_bad = pd.read_csv('./data/decayfre_bad_learners.csv')
# delta_data = pd.read_csv('./data/delta_data.csv')
# decay_data = pd.read_csv('./data/decay_data.csv')
# decayfre_data = pd.read_csv('./data/decayfre_data.csv')


# # let's see if decayfre is better than decay
# likelihood_ratio_test(decay_data, decayfre_data, df=1)
# bayes_factor(decay_data, decayfre_data)
# bayes_factor(delta_data, decay_data)

# # unlike delta versus decay, where decay is generally better than delta, decayfre is not always better than decay
# # although decayfre has a better model fit than decay, the difference lies in the huge BIC diff in some participants
# # in fact, more participants are better fitted by decay than decayfre
# diff = decay_data['BIC'] - delta_data['BIC']
# diff.mean()
# sum(diff < 0)

# best_beta = best_param_generator(decayfre_data, 'b')
# best_a = best_param_generator(decay_data, 'a')
# best_t = best_param_generator(decay_data, 't')
# pearsonr(propOptimal_CA['PropOptimal'], best_t)
#
# best_beta_good = best_param_generator(decayfre_good, 'b')
# best_a_good = best_param_generator(decay_good, 'a')
# best_t_good = best_param_generator(decay_good, 't')
# best_beta_bad = best_param_generator(decayfre_bad, 'b')
# best_a_bad = best_param_generator(decay_bad, 'a')
# best_t_bad = best_param_generator(decay_bad, 't')

# # combine the best fitting parameters into a dataframe
# best_param = pd.DataFrame({'best_beta_good': best_beta_good,
#                             'best_beta_bad': best_beta_bad,
#                             'best_a_good': best_a_good,
#                             'best_a_bad': best_a_bad,
#                             'best_t_good': best_t_good,
#                             'best_t_bad': best_t_bad})
# best_param.to_csv('./data/best_param.csv', index=False)

# ttest_ind(best_beta_good, best_beta_bad)
# ttest_ind(best_a_good, best_a_bad)
# ttest_ind(best_t_good, best_t_bad)

# print(decay_good['AIC'].mean())
# print(decayfre_good['AIC'].mean())
# print(decay_good['BIC'].mean())
# print(decayfre_good['BIC'].mean())

# reindex the subnum
data = data.reset_index(drop=True)
data.iloc[:, 1] = (data.index // 250) + 1
data['KeyResponse'] = data['KeyResponse'] - 1

# copy the index of all participants assigned to 3
bad_learners = assignment[assignment['assignments'] == 3].index.tolist()
bad_learners = [x + 1 for x in bad_learners]

good_learners = assignment[assignment['assignments'] == 1].index.tolist()
good_learners = [x + 1 for x in good_learners]

# keep only the participants assigned to bad learners
# bad_learners_data = data[data['Subnum'].isin(bad_learners)]
# bad_learners_data = bad_learners_data[bad_learners_data['SetSeen.'].isin([0, 1, 2])]
# bad_learners_data = bad_learners_data[bad_learners_data['Condition'] == 'Gains']

# keep only the participants assigned to good learners
# good_learners_data = data[data['Subnum'].isin(good_learners)]
# good_learners_data = good_learners_data[good_learners_data['SetSeen.'].isin([0, 1, 2])]
# good_learners_data = good_learners_data[good_learners_data['Condition'] == 'Gains']
# print(good_learners_data['SetSeen.'].unique())

# convert into dictionary
data_dict = dict_generator(data)
# bad_learners_dict = dict_generator(bad_learners_data)
# good_learners_dict = dict_generator(good_learners_data)

# set up the reward structure
reward_means = [0.65, 0.35, 0.75, 0.25]
reward_sd = [0.43, 0.43, 0.43, 0.43]

alt_reward_means = [-0.35, -0.65, -0.25, -0.75]

# fit the data
model_delta = ComputationalModels(reward_means, reward_sd,
                                    model_type='delta', condition='Both', num_trials=250)

model_decayfre = ComputationalModels(reward_means, reward_sd,
                                     model_type='decay_fre', condition='Both', num_trials=250)

model_decay = ComputationalModels(reward_means, reward_sd,
                                  model_type='decay', condition='Both', num_trials=250)

model_decay_choice = ComputationalModels(reward_means, reward_sd,
                                            model_type='decay_choice', condition='Both', num_trials=250)

model_decay_win = ComputationalModels(reward_means, reward_sd,
                                        model_type='decay_win', condition='Both', num_trials=250)

model_sampler_decay = ComputationalModels(reward_means, reward_sd,
                                          model_type='sampler_decay', condition='Both', num_trials=250, num_params=3)


# fit the model with all participants
results_data_decay_win = model_decay_win.fit(data_dict, num_iterations=100)
results_data_decay_win = pd.DataFrame(results_data_decay_win)
results_data_decay_win.iloc[:, 3] = results_data_decay_win.iloc[:, 3].astype(str)
results_data_decay_win.to_csv('./data/decay_win_gainloss.csv', index=False)

print(results_data_decay_win['AIC'].mean())
print(results_data_decay_win['BIC'].mean())

# results_good = model_sampler_decay.fit(good_learners_dict, num_iterations=100)
#
# result_good = pd.DataFrame(results_good)
# result_good.iloc[:, 3] = result_good.iloc[:, 3].astype(str)
# result_good.to_csv('./data/sampler_decay3_good_learners.csv', index=False)
#
# # sum up the AIC column
# print(result_good['AIC'].mean())
# print(result_good['BIC'].mean())
#
# results_bad = model_sampler_decay.fit(bad_learners_dict, num_iterations=100)
#
# result_bad = pd.DataFrame(results_bad)
# result_bad.iloc[:, 3] = result_bad.iloc[:, 3].astype(str)
# result_bad.to_csv('./data/sampler_decay3_bad_learners.csv', index=False)
#
# # sum up the AIC column
# print(result_bad['AIC'].mean())
# print(result_bad['BIC'].mean())

# # extract the best beta
# best_t_good = best_param_generator(result_good, 't')
# best_alpha_good = best_param_generator(result_good, 'a')
# best_beta_good = best_param_generator(result_good, 'b')
# best_t_bad = best_param_generator(result_bad, 't')
# best_alpha_bad = best_param_generator(result_bad, 'a')
# best_beta_bad = best_param_generator(result_bad, 'b')
#
# # best_t_good = best_param_generator(decay_good, 't')
# # best_alpha_good = best_param_generator(decay_good, 'a')
# # # best_beta_good = best_param_generator(decay_good, 'b')
# # best_t_bad = best_param_generator(decay_bad, 't')
# # best_alpha_bad = best_param_generator(decay_bad, 'a')
# # # best_beta_bad = best_param_generator(decay_bad, 'b')
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
#
# bad_learners_prob = assignment[assignment['assignments'] == 3]
# good_learners_prob = assignment[assignment['assignments'] == 1]
# pearsonr(best_t_bad, bad_learners_prob['prob3'])
# pearsonr(best_t_good, good_learners_prob['prob3'])

# # some simulations
# decay_choice = model_decay_win.simulate(AB_freq=100, CD_freq=50, num_iterations=1000)
#
# # unpacking the results
# all_data = []
#
# for res in decay_choice:
#     sim_num = res['simulation_num']
#     a_val = res['a']
#     t_val = res['t']
#     for trial_idx, trial_detail, ev in zip(res['trial_indices'], res['trial_details'], res['EV_history']):
#         data_row = {
#             'simulation_num': sim_num,
#             'trial_index': trial_idx,
#             'a': a_val,
#             't': t_val,
#             'pair': trial_detail['pair'],
#             'choice': trial_detail['choice'],
#             'reward': trial_detail['reward'],
#             'EV_A': ev[0],
#             'EV_B': ev[1],
#             'EV_C': ev[2],
#             'EV_D': ev[3]
#         }
#         all_data.append(data_row)
#
# df = pd.DataFrame(all_data)
#
# cols_to_mean = ['EV_A', 'EV_B', 'EV_C', 'EV_D']
# df_avg = df.groupby('trial_index')[cols_to_mean].mean().reset_index()
# plt.plot(df_avg['trial_index'], df_avg['EV_A'], label='EV_A')
# plt.plot(df_avg['trial_index'], df_avg['EV_B'], label='EV_B')
# plt.plot(df_avg['trial_index'], df_avg['EV_C'], label='EV_C')
# plt.plot(df_avg['trial_index'], df_avg['EV_D'], label='EV_D')
# plt.legend()
# plt.title('Decay Choice')
# plt.show()
