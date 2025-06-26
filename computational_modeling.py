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
    model_delta = ComputationalModels(model_type='delta')
    model_delta_PVL = ComputationalModels(model_type='delta_PVL')
    model_delta_PVL_relative = ComputationalModels(model_type='delta_PVL_relative')
    model_decay = ComputationalModels(model_type='decay')
    model_decay_win = ComputationalModels(model_type='decay_win')
    model_decay_PVL = ComputationalModels(model_type='decay_PVL')
    model_decay_PVL_relative = ComputationalModels(model_type='decay_PVL_relative')
    model_decay_PVPE = ComputationalModels(model_type='decay_PVPE')
    model_dual = DualProcessModel()

    model_list = [model_delta, model_delta_PVL, model_delta_PVL_relative,
                  model_decay, model_decay_win, model_decay_PVL, model_decay_PVL_relative, model_decay_PVPE,
                  model_dual]

    n_iterations = 100

    # test the data
    # test_data = E2_data[E2_data['Subnum'] == 1]
    # test_data = E1_data[(E1_data['Subnum'] >= 1) & (E1_data['Subnum'] <= 4)]
    # test_dict = dict_generator(test_data)
    #
    # testing_results = model_dual.fit(test_dict, 'Dual_Process', Gau_fun='Naive_Recency', Dir_fun='Linear_Recency',
    #                                     weight_Dir='softmax', weight_Gau='softmax', num_training_trials=120, num_exp_restart=200,
    #                                     num_iterations=1, initial_mode='first_trial_no_alpha')
    # testing_results = model_delta.fit(test_dict, num_training_trials=120, num_exp_restart=200, num_iterations=1, initial_mode='first_trial_no_alpha')

    # delta_results = model_delta_PVL.fit(E1_dict, num_training_trials=150, num_iterations=100, initial_mode='first_trial')
    # delta_results.to_csv('./data/ModelFitting/E1/delta_PVL_results.csv', index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # Experiment 1
    # ------------------------------------------------------------------------------------------------------------------
    # Fit all data (Since E1 is between-subjects, we can fit all data together)
    model_names = ['delta', 'delta_PVL', 'delta_PVL_relative', 'decay', 'decay_win', 'decay_PVL',
                   'decay_PVL_relative', 'decay_PVPE', 'dual']

    for i, model in enumerate(model_list):
        print(f"Fitting model: {model_names[i]}")
        save_dir = f'./data/ModelFitting/E1/{model_names[i]}_results.csv'
        # Check if the file already exists
        try:
         existing_results = pd.read_csv(save_dir)
         if not existing_results.empty:
              print(f"File {save_dir} already exists. Skipping model fitting.")
              continue
        except FileNotFoundError:
         pass

        if model_names[i] == 'dual':
            # Fit the dual-process model
            model_results = model.fit(E1_dict, 'Dual_Process', Gau_fun='Naive_Recency', Dir_fun='Linear_Recency',
                                      weight_Dir='softmax', weight_Gau='softmax', num_training_trials=150,
                                      num_iterations=n_iterations, num_exp_restart=250, initial_mode='first_trial_no_alpha')
        else:
        # For other models, fit them directly
            model_results = model.fit(E1_dict, num_training_trials=150, num_exp_restart=250, num_iterations=n_iterations,
                                      initial_mode='first_trial_no_alpha')

        model_results.to_csv(save_dir, index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # Experiment 2
    # ------------------------------------------------------------------------------------------------------------------
    # Fit all data (Here we manually force the model to set at trial 200 to account for task switching)
    for i, model in enumerate(model_list):
            save_dir = f'./data/ModelFitting/E2/{model_names[i]}_results.csv'
            # Check if the file already exists
            try:
             existing_results = pd.read_csv(save_dir)
             if not existing_results.empty:
                  print(f"File {save_dir} already exists. Skipping model fitting.")
                  continue
            except FileNotFoundError:
             pass

            if model_names[i] == 'dual':
                # Fit the dual-process model
                model_results = model.fit(E2_dict, 'Dual_Process', Gau_fun='Naive_Recency', Dir_fun='Linear_Recency',
                                          weight_Dir='softmax', weight_Gau='softmax', num_training_trials=120,
                                          num_exp_restart=200, num_iterations=n_iterations, initial_mode='first_trial_no_alpha')
            else:
            # For other models, fit them directly
                model_results = model.fit(E2_dict, num_training_trials=120, num_exp_restart=200,
                                          num_iterations=n_iterations, initial_mode='first_trial_no_alpha')

            model_results.to_csv(save_dir, index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # for baseline data
    # ------------------------------------------------------------------------------------------------------------------
    for i, model in enumerate(model_list):
        print(f'Fitting model: {model_names[i]}')
        save_dir = f'./data/ModelFitting/E2/Baseline/{model_names[i]}_results.csv'
        # Check if the file already exists
        try:
         existing_results = pd.read_csv(save_dir)
         if not existing_results.empty:
              print(f"File {save_dir} already exists. Skipping model fitting.")
              continue
        except FileNotFoundError:
         pass

        if model_names[i] == 'dual':
            # Fit the dual-process model
            model_results = model.fit(E2_baseline_dict, 'Dual_Process', Gau_fun='Naive_Recency', Dir_fun='Linear_Recency',
                                      weight_Dir='softmax', weight_Gau='softmax', num_training_trials=120,
                                      num_iterations=n_iterations, num_exp_restart=200, initial_mode='first_trial_no_alpha')
        else:
        # For other models, fit them directly
            model_results = model.fit(E2_baseline_dict, num_training_trials=120, num_iterations=n_iterations,
                                      num_exp_restart=200, initial_mode='first_trial_no_alpha')

        model_results.to_csv(save_dir, index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # for frequency data
    # ------------------------------------------------------------------------------------------------------------------
    for i, model in enumerate(model_list):
        print(f'Fitting model: {model_names[i]}')
        save_dir = f'./data/ModelFitting/E2/Frequency/{model_names[i]}_results.csv'
        # Check if the file already exists
        try:
         existing_results = pd.read_csv(save_dir)
         if not existing_results.empty:
              print(f"File {save_dir} already exists. Skipping model fitting.")
              continue
        except FileNotFoundError:
         pass

        if model_names[i] == 'dual':
            # Fit the dual-process model
            model_results = model.fit(E2_frequency_dict, 'Dual_Process', Gau_fun='Naive_Recency', Dir_fun='Linear_Recency',
                                      weight_Dir='softmax', weight_Gau='softmax', num_training_trials=120,
                                      num_iterations=n_iterations, num_exp_restart=200, initial_mode='first_trial_no_alpha')
        else:
        # For other models, fit them directly
            model_results = model.fit(E2_frequency_dict, num_training_trials=120, num_iterations=n_iterations,
                                      num_exp_restart=200, initial_mode='first_trial_no_alpha')

        model_results.to_csv(save_dir, index=False)


