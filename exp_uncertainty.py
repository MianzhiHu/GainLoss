import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities.utility_ComputationalModeling import (ComputationalModels, likelihood_ratio_test, dict_generator,
                                                     bayes_factor, best_param_generator)
from scipy.stats import pearsonr, spearmanr, ttest_ind
from utilities.utility_distribution import best_fitting_participants


# # first, we fit the data for the uncertainty study
# we have fitted the model with decayfre, decay, and delta, we can directly load the results
# overall, decay > delta > decayfre
# only in the CA condition in the Frequency-Uncertainty condition we see a significant correlation
uncertainty_decayfre = pd.read_csv('./data/uncertainty_decayfre.csv')
uncertainty_decayfre_neg = pd.read_csv('./data/uncertainty_decayfre_neg.csv')
uncertainty_decay = pd.read_csv('./data/uncertainty_decay.csv')
uncertainty_delta = pd.read_csv('./data/uncertainty_delta.csv')

# set up the reward structure
uncertainty_reward_means = [0.7, 0.3, 0.7, 0.3]
uncertainty_reward_sd = [0.43, 0.43, 0.12, 0.12]

# read in the data
uncertainty_data = pd.read_csv("C:/Users/zuire/PycharmProjects/Uncertainty_Personality/Data/full_data.csv")
uncertainty_prop_optimal = pd.read_csv("C:/Users/zuire/PycharmProjects/Uncertainty_Personality/Data/PropOptimal_Uncertainty.csv")
uncertainty_data['KeyResponse'] = uncertainty_data['KeyResponse'] - 1

# # comment or uncomment the following line to switch between the two conditions
# uncertainty_data = uncertainty_data[uncertainty_data['Condition'] == 'S2A1']

uncertainty_grouped = uncertainty_data.groupby('Subnum')

uncertainty_dict = dict_generator(uncertainty_data)

# fit the data
uncertainty_model_sampler_decay = ComputationalModels(uncertainty_reward_means, uncertainty_reward_sd,
                                                 model_type='sampler_decay', condition='Gains', num_params=3)
uncertainty_results_sampler_decay = uncertainty_model_sampler_decay.fit(uncertainty_dict, num_iterations=100)


# unpack the results
result = pd.DataFrame(uncertainty_results_sampler_decay)
result.iloc[:, 3] = result.iloc[:, 3].astype(str)
# sum up the AIC column
print(result['AIC'].mean())
print(result['BIC'].mean())
# save the results
result.to_csv('./data/uncertainty_sampler_decay_3.csv', index=False)


# extract the best beta
best_t = best_param_generator(result, 't')
best_alpha = best_param_generator(result, 'a')
best_beta = best_param_generator(result, 'b')

# proportion of optimal choices
uncertainty_prop_optimal_CA = uncertainty_prop_optimal[uncertainty_prop_optimal['ChoiceSet'] == 'CA']

pearsonr(uncertainty_prop_optimal_CA['PropOptimal'], best_alpha)

# likelihood_ratio_test(uncertainty_decay, uncertainty_decayfre_neg, df=1)
# bayes_factor(uncertainty_decay, uncertainty_decayfre_neg)


# # since this is a new model, we need to find the boundaries for beta
# uncertainty_model_decayfre = ComputationalModels(uncertainty_reward_means, uncertainty_reward_sd,
#                                                     model_type='decay_fre', condition='Gains')
#
# best_bounds = []
#
# # potential_beta_upper = [x for x in np.arange(0, 1., 0.1) if x != 0]
# # potential_beta_lower = [x for x in np.arange(-1, 0, 0.1) if x != 0]
# # potential_beta_lower.reverse()
#
# potential_beta_upper = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.1, 0.01]
# potential_beta_lower = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -0.5, -0.1, -0.01]
#
# potential_beta = list(zip(potential_beta_lower, potential_beta_upper))
#
# for beta_lower, beta_upper in potential_beta:
#     print(beta_lower, beta_upper)
#     best_bounds_results = uncertainty_model_decayfre.fit(uncertainty_dict, num_iterations=100,
#                                                                     beta_lower=beta_lower, beta_upper=beta_upper)
#
#     best_bounds_results = pd.DataFrame(best_bounds_results)
#     best_bounds_results.iloc[:, 3] = best_bounds_results.iloc[:, 3].astype(str)
#
#     # sum up the AIC column
#     aic = best_bounds_results['AIC'].mean()
#     bic = best_bounds_results['BIC'].mean()
#     beta = best_bounds_results['best_parameters'].apply(
#         lambda x: float(x.strip('[]').split()[2]) if isinstance(x, str) else np.nan
#     )
#     mean_beta = beta.mean()
#     sd_beta = beta.std()
#     max_beta = beta.max()
#     min_beta = beta.min()
#
#     best_bounds.append([beta_lower, beta_upper, aic, bic, mean_beta, sd_beta, max_beta, min_beta])
#
# best_bounds = pd.DataFrame(best_bounds, columns=['beta_lower', 'beta_upper', 'AIC', 'BIC', 'mean_beta', 'sd_beta',
#                                                  'max_beta', 'min_beta'])

# best_bounds.to_csv('./data/best_bounds.csv', index=False)

# # simulation starts here
# model = ComputationalModels(uncertainty_reward_means, uncertainty_reward_sd,
#                             model_type='sampler_decay', condition='Gains')
# results = model.simulate(AB_freq=100, CD_freq=50, num_iterations=1000)
#
#
# # unpacking the results
# all_data = []
#
# for res in results:
#     sim_num = res['simulation_num']
#     a_val = res['a']
#     b_val = res['b']
#     t_val = res['t']
#     for trial_idx, trial_detail, ev in zip(res['trial_indices'], res['trial_details'], res['EV_history']):
#         data_row = {
#             'simulation_num': sim_num,
#             'trial_index': trial_idx,
#             'a': a_val,
#             'b': b_val,
#             't': t_val,
#             'pair': trial_detail['pair'],
#             'choice': trial_detail['choice'],
#             'EV_A': ev[0],
#             'EV_B': ev[1],
#             'EV_C': ev[2],
#             'EV_D': ev[3]
#         }
#         all_data.append(data_row)
#
# df = pd.DataFrame(all_data)
#
# # # save the data
# # df.to_csv('./data/sim_decay_model.csv', index=False)
#
#
#
# # # Plotting the EVs over the trials
# # decay = pd.read_csv('./data/sim_decay_model.csv')
# # decayEF = pd.read_csv('./data/sim_decay_model_EF.csv')
# # decayfre = pd.read_csv('./data/sim_decayfre_model.csv')
# # decayfreEF = pd.read_csv('./data/sim_decayfre_model_EF.csv')
# # delta = pd.read_csv('./data/sim_delta_model.csv')
# # deltaEF = pd.read_csv('./data/sim_delta_model_EF.csv')
#
# cols_to_mean = ['EV_A', 'EV_B', 'EV_C', 'EV_D']
# df_avg = df.groupby('trial_index')[cols_to_mean].mean().reset_index()
# plt.plot(df_avg['trial_index'], df_avg['EV_A'], label='EV_A')
# plt.plot(df_avg['trial_index'], df_avg['EV_B'], label='EV_B')
# plt.plot(df_avg['trial_index'], df_avg['EV_C'], label='EV_C')
# plt.plot(df_avg['trial_index'], df_avg['EV_D'], label='EV_D')
# plt.legend()
# plt.show()


# sim_list = [delta, deltaEF, decay, decayEF, decayfre, decayfreEF]
#
# for sim in sim_list:
#     sim_avg = sim.groupby('trial_index')[cols_to_mean].mean().reset_index()
#     plt.plot(sim_avg['trial_index'], sim_avg['EV_A'], label='EV_A')
#     plt.plot(sim_avg['trial_index'], sim_avg['EV_B'], label='EV_B')
#     plt.plot(sim_avg['trial_index'], sim_avg['EV_C'], label='EV_C')
#     plt.plot(sim_avg['trial_index'], sim_avg['EV_D'], label='EV_D')
#     plt.legend()
#     plt.show()


