import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities.utility_ComputationalModeling import ComputationalModels, likelihood_ratio_test
from scipy.stats import pearsonr, spearmanr, ttest_ind
from utilities.utility_distribution import best_fitting_participants

# Read in the data
data = pd.read_csv('./data/ABCDGainsLossesData_F2023.csv')
assignment = pd.read_csv('./data/trimodal_assignments_CA.csv')

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

# copy the index of all participants assigned to 3
bad_learners = assignment[assignment['assignments'] == 3].index.tolist()
bad_learners = [x + 1 for x in bad_learners]

# keep only the participants assigned to bad learners
bad_learners_data = data[data['Subnum'].isin(bad_learners)]
bad_learners_data = bad_learners_data[bad_learners_data['SetSeen.'].isin([0, 1, 2])]
bad_learners_data['KeyResponse'] = bad_learners_data['KeyResponse'] - 1

# convert into dictionary
bad_learners_dict = {}

for name, group in bad_learners_data.groupby('Subnum'):
    bad_learners_dict[name] = {
        'reward': group['Reward'].tolist(),
        'choiceset': group['SetSeen.'].tolist(),
        'choice': group['KeyResponse'].tolist(),
    }

# set up the reward structure
reward_means = [0.65, 0.35, 0.75, 0.25]
reward_sd = [0.43, 0.43, 0.43, 0.43]

# fit the data
model_delta = ComputationalModels(reward_means, reward_sd,
                                     model_type='delta', condition='Both', num_trials=175)
results_delta = model_delta.fit(bad_learners_dict, num_iterations=1000)

result = pd.DataFrame(results_delta)
# # sum up the AIC column
print(result['AIC'].mean())