import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities.utility_ComputationalModeling import ComputationalModels, likelihood_ratio_test


# # first, we fit the data for the uncertainty study
# set up the reward structure
uncertainty_reward_means = [0.7, 0.3, 0.7, 0.3]
uncertainty_reward_sd = [0.43, 0.43, 0.12, 0.12]

# read in the data
uncertainty_data = pd.read_csv("C:/Users/zuire/PycharmProjects/Uncertainty_Personality/Data/full_data.csv")
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
uncertainty_model_decayfre = ComputationalModels(uncertainty_reward_means, uncertainty_reward_sd,
                                                 model_type='decay_fre', condition='Gains')
uncertainty_results_decayfre = uncertainty_model_decayfre.fit(uncertainty_dict, num_iterations=1000)

# likelihood_ratio_test(results_decay, results_decayfre, df=1)


# # simulation starts here
# model = ComputationalModels(reward_means, reward_sd, model_type='decay', condition='Gains')
# results = model.simulate(AB_freq=75, CD_freq=75, num_iterations=1000)
# #
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


