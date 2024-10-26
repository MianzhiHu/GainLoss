import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.ComputationalModeling import ComputationalModels, dict_generator
from utils.DualProcess import DualProcessModel
from scipy.stats import pearsonr, spearmanr, ttest_ind
from utilities.utility_distribution import best_fitting_participants

# Read in the data
data_gains = pd.read_csv('./data/data_gains.csv')
data = pd.read_csv('./data/data_id.csv')

# # test the data
# data = data[data['Subnum'] <= 1]

# reindex the subnum
data = data.reset_index(drop=True)
data['KeyResponse'] = data['KeyResponse'] - 1
data.rename(columns={'SetSeen ': 'SetSeen.'}, inplace=True)

data_gains['KeyResponse'] = data_gains['KeyResponse'] - 1

# divide by condition
baseline = data[data['Condition'] == 'Baseline']
frequency = data[data['Condition'] == 'Frequency']

gains_baseline = data_gains[data_gains['Condition'] == 'GainsEF']
gains_frequency = data_gains[data_gains['Condition'] == 'Gains']

# convert into dictionary
data_dict = dict_generator(data)
baseline_dict = dict_generator(baseline)
frequency_dict = dict_generator(frequency)

data_gains_dict = dict_generator(data_gains)
gains_baseline_dict = dict_generator(gains_baseline)
gains_frequency_dict = dict_generator(gains_frequency)

if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # Experiment 1
    # ------------------------------------------------------------------------------------------------------------------
    # define the model
    model_delta = ComputationalModels(model_type='delta', condition='Gains', num_trials=250)
    model_decay = ComputationalModels(model_type='decay', condition='Gains', num_trials=250)
    model_dual = DualProcessModel(num_trials=250)

    # # fit the model
    # # ------------------------------------------------------------------------------------------------------------------
    # # for all data
    # # ------------------------------------------------------------------------------------------------------------------
    # delta_results = model_delta.fit(data_gains_dict, num_iterations=200)
    # delta_results.to_csv('./data/ModelFitting/delta_gains.csv', index=False)
    #
    # decay_results = model_decay.fit(data_gains_dict, num_iterations=200)
    # decay_results.to_csv('./data/ModelFitting/decay_gains.csv', index=False)
    #
    # dual_results = model_dual.fit(data_gains_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency',
    #                               Dir_fun='Linear_Recency', weight_Dir='softmax', weight_Gau='softmax',
    #                               num_iterations=200)
    # dual_results.to_csv('./data/ModelFitting/dual_gains.csv', index=False)
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # # for baseline data
    # # ------------------------------------------------------------------------------------------------------------------
    # delta_baseline = model_delta.fit(gains_baseline_dict, num_iterations=200)
    # delta_baseline.to_csv('./data/ModelFitting/delta_baseline_gains.csv', index=False)
    #
    # decay_baseline = model_decay.fit(gains_baseline_dict, num_iterations=200)
    # decay_baseline.to_csv('./data/ModelFitting/decay_baseline_gains.csv', index=False)
    #
    # dual_baseline = model_dual.fit(gains_baseline_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency',
    #                                   Dir_fun='Linear_Recency', weight_Dir='softmax', weight_Gau='softmax',
    #                                   num_iterations=200)
    # dual_baseline.to_csv('./data/ModelFitting/dual_baseline_gains.csv', index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # for frequency data
    # ------------------------------------------------------------------------------------------------------------------
    delta_frequency = model_delta.fit(gains_frequency_dict, num_iterations=200)
    delta_frequency.to_csv('./data/ModelFitting/delta_frequency_gains.csv', index=False)

    decay_frequency = model_decay.fit(gains_frequency_dict, num_iterations=200)
    decay_frequency.to_csv('./data/ModelFitting/decay_frequency_gains.csv', index=False)

    dual_frequency = model_dual.fit(gains_frequency_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency',
                                        Dir_fun='Linear_Recency', weight_Dir='softmax', weight_Gau='softmax',
                                        num_iterations=200)
    dual_frequency.to_csv('./data/ModelFitting/dual_frequency_gains.csv', index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # Experiment 2
    # ------------------------------------------------------------------------------------------------------------------
    # define the model
    # model_delta = ComputationalModels(model_type='delta', condition='Gains', num_trials=200)
    # model_decay = ComputationalModels(model_type='decay', condition='Gains', num_trials=200)
    # model_dual = DualProcessModel(num_trials=200)

    # fit the model
    # ------------------------------------------------------------------------------------------------------------------
    # for all data
    # ------------------------------------------------------------------------------------------------------------------
    # delta_results = model_delta.fit(data_dict, num_iterations=200)
    # delta_results.to_csv('./data/ModelFitting/delta_id.csv', index=False)
    #
    # decay_results = model_decay.fit(data_dict, num_iterations=200)
    # decay_results.to_csv('./data/ModelFitting/decay_id.csv', index=False)
    #
    # dual_results = model_dual.fit(data_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency',
    #                               Dir_fun='Linear_Recency', weight_Dir='softmax', weight_Gau='softmax',
    #                               num_iterations=200)
    # dual_results.to_csv('./data/ModelFitting/dual_id.csv', index=False)

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


