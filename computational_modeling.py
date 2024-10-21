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

# reindex the subnum
data = data.reset_index(drop=True)
data['KeyResponse'] = data['KeyResponse'] - 1
data.rename(columns={'SetSeen ': 'SetSeen.'}, inplace=True)

# convert into dictionary
data_dict = dict_generator(data)

if __name__ == '__main__':
    # define the model
    model_delta = ComputationalModels(model_type='delta', condition='Gains', num_trials=200)
    model_decay = ComputationalModels(model_type='decay', condition='Gains', num_trials=200)
    model_dual = DualProcessModel(num_trials=200)

    # fit the model
    delta_results = model_delta.fit(data_dict, num_iterations=200)
    delta_results.to_csv('./data/ModelFitting/delta_id.csv', index=False)

    decay_results = model_decay.fit(data_dict, num_iterations=200)
    decay_results.to_csv('./data/ModelFitting/decay_id.csv', index=False)

    dual_results = model_dual.fit(data_dict, 'Entropy_Dis_ID', Gau_fun='Naive_Recency',
                                  Dir_fun='Linear_Recency', weight_Dir='softmax', weight_Gau='softmax',
                                  num_iterations=200)
    dual_results.to_csv('./data/ModelFitting/dual_id.csv', index=False)
