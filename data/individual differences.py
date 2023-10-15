import pandas as pd
import scipy.stats as stats
from utility import correlation_test

# Read in the data
data = pd.read_csv('C:/Users/zuire/PycharmProjects/GainLoss/data/data.csv')

# test all the correlations

# trials with significant correlations (PropOptimal):
# AB: Losses, LossesEF, Gains
# CD: Losses, Gains
# CA: LossesEF
# CB: Losses, Gains
# AD: Losses, LossesEF
# BD: LossesEF, GainsEF
# trials with significant correlations (RT):
# AB: Losses
# CD: LossesEF, Gains
# CA: Losses, LossesEF
# CB: Losses, LossesEF
# AD: Losses, LossesEF
# BD: Losses, LossesEF

trial_list = ['AB', 'CD', 'CA', 'CB', 'AD', 'BD']
condition_list = ['Losses', 'LossesEF', 'Gains', 'GainsEF']
loss_list = ['Losses', 'LossesEF']
variable_of_interest_list = ['RT']

for trial in trial_list:
    for condition in condition_list:
        for variable_of_interest in variable_of_interest_list:
            correlation_test(data, trial, condition, variable_of_interest, checker=True)


correlation_test(data, 'BD', 'GainsEF', 'PropOptimal', sig_only=True)