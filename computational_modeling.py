import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.ComputationalModeling import ComputationalModels, dict_generator
from utils.DualProcess import DualProcessModel
from scipy.stats import pearsonr, spearmanr, ttest_ind
from utilities.utility_distribution import best_fitting_participants

# Read in the data
data = pd.read_csv('./data/data_id.csv')

# # test the data
# data = data[data['Subnum'] <= 1]

# reindex the subnum
data = data.reset_index(drop=True)
data['KeyResponse'] = data['KeyResponse'] - 1
data.rename(columns={'SetSeen ': 'SetSeen.'}, inplace=True)

# divide by condition
baseline = data[data['Condition'] == 'Baseline']
frequency = data[data['Condition'] == 'Frequency']

# convert into dictionary
data_dict = dict_generator(data)
baseline_dict = dict_generator(baseline)
frequency_dict = dict_generator(frequency)

if __name__ == '__main__':
    # define the model
    model_delta = ComputationalModels(model_type='delta', condition='Gains', num_trials=200)
    model_decay = ComputationalModels(model_type='decay', condition='Gains', num_trials=200)
    model_dual = DualProcessModel(num_trials=200)

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

    # ------------------------------------------------------------------------------------------------------------------
    # for baseline data
    # ------------------------------------------------------------------------------------------------------------------
    delta_baseline = model_delta.fit(baseline_dict, num_iterations=200)
    delta_baseline.to_csv('./data/ModelFitting/delta_baseline_id.csv', index=False)

    decay_baseline = model_decay.fit(baseline_dict, num_iterations=200)
    decay_baseline.to_csv('./data/ModelFitting/decay_baseline_id.csv', index=False)

    dual_baseline = model_dual.fit(baseline_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency',
                                      Dir_fun='Linear_Recency', weight_Dir='softmax', weight_Gau='softmax',
                                      num_iterations=200)
    dual_baseline.to_csv('./data/ModelFitting/dual_baseline_id.csv', index=False)

    print(f'Baseline Delta: {delta_baseline["AIC"].mean()}')
    print(f'Baseline Decay: {decay_baseline["AIC"].mean()}')
    print(f'Baseline Dual: {dual_baseline["AIC"].mean()}')

    print(f'Baseline Delta: {delta_baseline["BIC"].mean()}')
    print(f'Baseline Decay: {decay_baseline["BIC"].mean()}')
    print(f'Baseline Dual: {dual_baseline["BIC"].mean()}')
    # ------------------------------------------------------------------------------------------------------------------
    # for frequency data
    # ------------------------------------------------------------------------------------------------------------------
    delta_frequency = model_delta.fit(frequency_dict, num_iterations=200)
    delta_frequency.to_csv('./data/ModelFitting/delta_frequency_id.csv', index=False)

    decay_frequency = model_decay.fit(frequency_dict, num_iterations=200)
    decay_frequency.to_csv('./data/ModelFitting/decay_frequency_id.csv', index=False)

    dual_frequency = model_dual.fit(frequency_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency',
                                        Dir_fun='Linear_Recency', weight_Dir='softmax', weight_Gau='softmax',
                                        num_iterations=200)
    dual_frequency.to_csv('./data/ModelFitting/dual_frequency_id.csv', index=False)

    # print(f'AIC delta: {delta_results["AIC"].mean()}')
    # print(f'AIC decay: {decay_results["AIC"].mean()}')
    # print(f'AIC dual: {dual_results["AIC"].mean()}')
    # print(f'BIC delta: {delta_results["BIC"].mean()}')
    # print(f'BIC decay: {decay_results["BIC"].mean()}')
    # print(f'BIC dual: {dual_results["BIC"].mean()}')
