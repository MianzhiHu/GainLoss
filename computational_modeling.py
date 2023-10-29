import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities.utility_ComputationalModeling import (ComputationalModels, likelihood_ratio_test,
                                                     unpacking_results)
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

uncertainty_grouped = uncertainty_data.groupby('Subnum')

uncertainty_dict = {}

for subnum, group in uncertainty_grouped:
    uncertainty_dict[subnum] = {
        'reward': group['Reward'].tolist(),
        'choiceset': group['SetSeen.'].tolist(),
        'choice': group['KeyResponse'].tolist(),
    }


# fit the data
uncertainty_model_neg = ComputationalModels(uncertainty_reward_means, uncertainty_reward_sd,
                                                 model_type='decay_fre', condition='Gains')
uncertainty_results_neg = uncertainty_model_neg.fit(uncertainty_dict, num_iterations=1000, beta_lower=-2, beta_upper=2)


# # unpack the results
# uncertainty_results_neg_df = unpacking_results(uncertainty_results_neg)
# uncertainty_decayfre_t = unpacking_results(uncertainty_results_t)
# uncertainty_results_decay = unpacking_results(uncertainty_results_decayfre)
# uncertainty_decayfre_t = unpacking_results(uncertainty_results_t)
#
# results_df = pd.DataFrame(uncertainty_results_delta)
# results_df['AIC'] = results_df['AIC'].fillna(results_df['AIC'].max())
# results_df['BIC'] = results_df['BIC'].fillna(results_df['BIC'].max())
# results_df = results_df.iloc[:-1]

# # save the results
# results_df.to_csv('./data/uncertainty_delta.csv', index=False)


# # unpack the best fitting beta
# beta_best = uncertainty_decayfre_neg['best_parameters'].apply(
#     lambda x: float(x.strip('[]').split()[2]) if isinstance(x, str) else np.nan
# )
#
# # calculate the proportion of participants who are best fitted by decayfre
# # decay > delta > decayfre_neg > decayfre
# best_fitting_participants(uncertainty_decayfre_neg, uncertainty_decayfre, uncertainty_decay, uncertainty_delta)
# # decay > delta > decayfre_neg
# best_fitting_participants(uncertainty_decayfre_neg, uncertainty_decay, uncertainty_delta)
# # decay > decayfre_neg
# best_decayfre = best_fitting_participants(uncertainty_decayfre_neg, uncertainty_decay, p_index=0)
# best_decay = best_fitting_participants(uncertainty_decayfre_neg, uncertainty_decay, p_index=1)
# # delta > decayfre_neg (slightly)
# best_fitting_participants(uncertainty_decayfre_neg, uncertainty_delta)
#
# best_decayfre_data = uncertainty_data[uncertainty_data['Subnum'].isin(best_decayfre)]
# best_decay_data = uncertainty_data[uncertainty_data['Subnum'].isin(best_decay)]
#
# best_decayfre_betas = [beta_best[p - 1] for p in best_decayfre]
# best_decay_betas = [beta_best[p - 1] for p in best_decay]
#
# uncertainty_A1 = uncertainty_data[uncertainty_data['Condition'] == 'S2A1']
# uncertainty_A2 = uncertainty_data[uncertainty_data['Condition'] == 'S2A2']
#
# # find the participant number
# A1_subnum = uncertainty_A1['Subnum'].unique().tolist()
# A2_subnum = uncertainty_A2['Subnum'].unique().tolist()
#
# A1_decayfre = [p for p in A1_subnum if p in best_decayfre]
#
# A1_selected_betas = [beta_best[p - 1] for p in A1_subnum]
# A1_decayfre_betas = [beta_best[p - 1] for p in A1_decayfre]
#
#
# A2_selected_betas = [beta_best[p - 1] for p in A2_subnum]
#
# A1_propOptimal = uncertainty_prop_optimal.loc[uncertainty_prop_optimal['Subnum'].isin(A1_subnum)]
# A2_propOptimal = uncertainty_prop_optimal.loc[uncertainty_prop_optimal['Subnum'].isin(A2_subnum)]
# decayfre_propOptimal = uncertainty_prop_optimal.loc[uncertainty_prop_optimal['Subnum'].isin(best_decayfre)]
# decay_propOptimal = uncertainty_prop_optimal.loc[uncertainty_prop_optimal['Subnum'].isin(best_decay)]
# A1_decayfre_propOptimal = A1_propOptimal.loc[A1_propOptimal['Subnum'].isin(best_decayfre)]
# A1_decayfre_propOptimal_CA = A1_decayfre_propOptimal[A1_decayfre_propOptimal['ChoiceSet'] == 'CA']
#
# # first calculate the mean
# A1_propOptimal_mean = A1_propOptimal.groupby('Subnum')['PropOptimal'].mean().reset_index()
# A1_propOptimal_CA = A1_propOptimal[A1_propOptimal['ChoiceSet'] == 'CA']
# A1_propOptimal_BD = A1_propOptimal[A1_propOptimal['ChoiceSet'] == 'BD']
#
# A2_propOptimal_mean = A2_propOptimal.groupby('Subnum')['PropOptimal'].mean().reset_index()
# A2_propOptimal_CA = A2_propOptimal[A2_propOptimal['ChoiceSet'] == 'CA']
# decayfre_propOptimal_CA = decayfre_propOptimal[decayfre_propOptimal['ChoiceSet'] == 'CA']
# decay_propOptimal_CA = decay_propOptimal[decay_propOptimal['ChoiceSet'] == 'CA']
# decayfre_propOptimal_BD = decayfre_propOptimal[decayfre_propOptimal['ChoiceSet'] == 'BD']
# decay_propOptimal_BD = decay_propOptimal[decay_propOptimal['ChoiceSet'] == 'BD']
#
#
# # test the correlation
# print(pearsonr(A1_propOptimal_mean['PropOptimal'], A1_selected_betas))
#
#
# # test the correlation between beta and every personality trait
# for column in uncertainty_data.columns[4:20]:
#     personality = uncertainty_grouped[column].first().tolist()
#     print(column, pearsonr(beta_best, personality))
#
# for column in best_decayfre_data.columns[4:20]:
#     personality = best_decayfre_data.groupby('Subnum')[column].first().tolist()
#     print(column, pearsonr(best_decayfre_betas, personality))
#
#
# # conduct t-test for two groups
# # Group by 'Subnum' and compute the mean for all columns
# cols_to_select = [0] + list(range(4, 20))
# grouped_means_decayfre = best_decayfre_data.iloc[:, cols_to_select].groupby('Subnum').mean()
# grouped_means_decay = best_decay_data.iloc[:, cols_to_select].groupby('Subnum').mean()
#
# # Conduct t-test
# for column in grouped_means_decayfre.columns:
#     print(column, ttest_ind(grouped_means_decayfre[column], grouped_means_decay[column]))
#
#
# # draw a scatter plot
# plt.scatter(A1_decayfre_betas, A1_decayfre_propOptimal_CA['PropOptimal'])
# slope, intercept = np.polyfit(best_decayfre_betas, decayfre_propOptimal_CA['PropOptimal'], 1)
# plt.plot(best_decayfre_betas, slope * np.array(best_decayfre_betas) + intercept, color='red',
#          label=f"y = {slope:.3f}x + {intercept:.3f}")
#
# plt.xlabel('Beta')
# plt.ylabel('PropOptimal')
# plt.show()
#
# plt.hist(decayfre_propOptimal_CA['PropOptimal'], density=True, alpha=0.6, color='g')
# plt.xlabel('PropOptimal')
# plt.ylabel('Density')
# plt.show()


# likelihood_ratio_test(results_decay, results_decayfre, df=1)




# # since this is a new model, we need to find the boundaries for beta
# uncertainty_model_decayfre = ComputationalModels(uncertainty_reward_means, uncertainty_reward_sd,
#                                                     model_type='decay_fre', condition='Gains')
#
# nll_best_beta = 100000
# best_beta_lower = None
# best_beta_upper = None
#
# for i in range(100000):
#     print(i)
#     beta_lower = np.random.uniform(-100, 0)
#     beta_upper = np.random.uniform(0, 100)
#     uncertainty_results_decayfre = uncertainty_model_decayfre.fit(uncertainty_dict, num_iterations=100,
#                                                                     beta_lower=beta_lower, beta_upper=beta_upper)
#     nll = uncertainty_results_decayfre[186]['total_nll']
#
#     if nll < nll_best_beta:
#         nll_best_beta = nll
#         best_beta_lower = beta_lower
#         best_beta_upper = beta_upper
#
# print(best_beta_lower, best_beta_upper)


# # simulation starts here
# model = ComputationalModels(uncertainty_reward_means, uncertainty_reward_sd, model_type='decay_fre', condition='Gains')
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


