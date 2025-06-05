import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.ComputationalModeling import ComputationalModels, dict_generator
from utils.DualProcess import DualProcessModel
from scipy.stats import pearsonr, spearmanr, ttest_ind
from utilities.utility_distribution import best_fitting_participants

# Read in the data
E1_data = pd.read_csv('./data/data_gains.csv')
E2_data = pd.read_csv('./data/data_id.csv')

# Some data cleaning
E1_data['KeyResponse'] = E1_data['KeyResponse'] - 1

E2_data = E2_data.reset_index(drop=True)
E2_data['KeyResponse'] = E2_data['KeyResponse'] - 1

# Divide by condition
E1_baseline = E1_data[E1_data['Condition'] == 'GainsEF']
E1_frequency = E1_data[E1_data['Condition'] == 'Gains']

E2_baseline = E2_data[E2_data['Condition'] == 'Baseline']
E2_frequency = E2_data[E2_data['Condition'] == 'Frequency']

# convert into dictionary
E1_dict = dict_generator(E1_data)

E2_dict = dict_generator(E2_data)
E2_baseline_dict = dict_generator(E2_baseline)
E2_frequency_dict = dict_generator(E2_frequency)

if __name__ == '__main__':
    # Define the model
    model_delta = ComputationalModels(model_type='delta', condition='Gains')
    model_delta_PVL = ComputationalModels(model_type='delta_PVL', condition='Gains')
    model_decay = ComputationalModels(model_type='decay', condition='Gains')
    model_decay_win = ComputationalModels(model_type='decay_win', condition='Gains')
    model_dual = DualProcessModel()

    n_iterations = 100

    # # test the data
    # test_data = E1_data[E1_data['Subnum'] <= 1]
    # test_dict = dict_generator(test_data)
    #
    # testing_results = model_dual.fit(test_dict, 'Dual_Process', Gau_fun='Naive_Recency', Dir_fun='Linear_Recency',
    #                                     weight_Dir='softmax', weight_Gau='softmax', num_iterations=1)
    # ------------------------------------------------------------------------------------------------------------------
    # Experiment 1
    # ------------------------------------------------------------------------------------------------------------------
    # # Fit all data (Since E1 is between-subjects, we can fit all data together)
    # model_names = ['delta', 'delta_PVL', 'decay', 'decay_win', 'dual']
    #
    # for i, model in enumerate([model_delta, model_delta_PVL, model_decay, model_decay_win, model_dual]):
    #         save_dir = f'./data/ModelFitting/E1/{model_names[i]}_results.csv'
    #         # Check if the file already exists
    #         try:
    #          existing_results = pd.read_csv(save_dir)
    #          if not existing_results.empty:
    #               print(f"File {save_dir} already exists. Skipping model fitting.")
    #               continue
    #         except FileNotFoundError:
    #          pass
    #
    #         if model_names[i] == 'dual':
    #             # Fit the dual-process model
    #             model_results = model.fit(E1_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency', Dir_fun='Linear_Recency',
    #                                       weight_Dir='softmax', weight_Gau='softmax', num_training_trials=150,
    #                                       num_iterations=n_iterations)
    #         else:
    #         # For other models, fit them directly
    #             model_results = model.fit(E1_dict, num_training_trials=150, num_iterations=n_iterations)
    #
    #         model_results.to_csv(save_dir, index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # Experiment 2
    # ------------------------------------------------------------------------------------------------------------------
    # # Fit all data (Here we manually force the model to set at trial 200 to account for task switching)
    # model_names = ['delta', 'delta_PVL', 'decay', 'decay_win', 'dual']
    #
    # for i, model in enumerate([model_delta, model_delta_PVL, model_decay, model_decay_win, model_dual]):
    #         save_dir = f'./data/ModelFitting/E2/{model_names[i]}_results.csv'
    #         # Check if the file already exists
    #         try:
    #          existing_results = pd.read_csv(save_dir)
    #          if not existing_results.empty:
    #               print(f"File {save_dir} already exists. Skipping model fitting.")
    #               continue
    #         except FileNotFoundError:
    #          pass
    #
    #         if model.model_type == 'dual':
    #             # Fit the dual-process model
    #             model_results = model.fit(E2_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency', Dir_fun='Linear_Recency',
    #                                       weight_Dir='softmax', weight_Gau='softmax', num_iterations=n_iterations)
    #         else:
    #         # For other models, fit them directly
    #             model_results = model.fit(E2_dict, num_training_trials=120, num_iterations=n_iterations)
    #
    #         model_results.to_csv(save_dir, index=False)
    # # ------------------------------------------------------------------------------------------------------------------
    # # for baseline data
    # # ------------------------------------------------------------------------------------------------------------------
    # delta_baseline = model_delta.fit(baseline_dict, num_iterations=200)
    # delta_baseline.to_csv('./data/ModelFitting/delta_baseline_id.csv', index=False)
    #
    # decay_baseline = model_decay.fit(baseline_dict, num_iterations=200)
    # decay_baseline.to_csv('./data/ModelFitting/decay_baseline_id.csv', index=False)
    #
    # dual_baseline = model_dual.fit(baseline_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency',
    #                                   Dir_fun='Linear_Recency', weight_Dir='softmax', weight_Gau='softmax',
    #                                   num_iterations=200)
    # dual_baseline.to_csv('./data/ModelFitting/dual_baseline_id.csv', index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # for frequency data
    # ------------------------------------------------------------------------------------------------------------------
    # delta_frequency = model_delta.fit(frequency_dict, num_iterations=200)
    # delta_frequency.to_csv('./data/ModelFitting/delta_frequency_id.csv', index=False)
    #
    # decay_frequency = model_decay.fit(frequency_dict, num_iterations=200)
    # decay_frequency.to_csv('./data/ModelFitting/decay_frequency_id.csv', index=False)

    # dual_frequency = model_dual.fit(frequency_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency',
    #                                     Dir_fun='Linear_Recency', weight_Dir='softmax', weight_Gau='softmax',
    #                                     num_iterations=200)
    # dual_frequency.to_csv('./data/ModelFitting/dual_frequency_id.csv', index=False)


