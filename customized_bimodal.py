import numpy as np
import pandas as pd
from scipy.stats import norm
from utilities.utility_model import em_model, e_step, m_step, log_likelihood


# Read in the data
data = pd.read_csv('./data/data.csv')

# remove the outlier trials (RT > 10s)
data = data[data['RT'] < 10000]

data.loc[data['PropOptimal'] == 1, 'PropOptimal'] = 0.999
data.loc[data['PropOptimal'] == 0, 'PropOptimal'] = 0.001
# Subset the data
ABoptimal = data[data['ChoiceSet'] == 'AB']['PropOptimal']
CDoptimal = data[data['ChoiceSet'] == 'CD']['PropOptimal']
CAoptimal = data[data['ChoiceSet'] == 'CA']['PropOptimal']
BDoptimal = data[data['ChoiceSet'] == 'BD']['PropOptimal']
CBoptimal = data[data['ChoiceSet'] == 'CB']['PropOptimal']
ADoptimal = data[data['ChoiceSet'] == 'AD']['PropOptimal']

# now fit the em model to each of the six choice sets
trial_list = [ABoptimal, CDoptimal, CAoptimal, BDoptimal, CBoptimal, ADoptimal]
condition_list = ['Losses', 'LossesEF', 'Gains', 'GainsEF']




# Starting parameter estimates
mu1, sd1 = 0.25, 0.2
mu2, sd2 = 0.75, 0.2
ppi = 0.5

# Assuming your data is stored in a list or numpy array named dat
tolerance = 0.00001
change = np.inf
oldppi = 0

while change > tolerance:
    # E-Step
    resp = e_step(ADoptimal, mu1, mu2, sd1, sd2, ppi)
    # M-Step
    mu1, mu2, sd1, sd2, newppi = m_step(ADoptimal, resp)

    change = np.abs(newppi - oldppi)
    oldppi = newppi
    ppi = newppi

print("Converged Parameters:")
print(f"Mu1: {mu1}, SD1: {sd1}")
print(f"Mu2: {mu2}, SD2: {sd2}")
print(f"Proportion for distribution 2 (ppi): {ppi}")

LL = log_likelihood(ADoptimal, mu1, mu2, sd1, sd2, ppi)

mu_null, sd_null = norm.fit(ADoptimal)
LL_null = log_likelihood(ADoptimal, mu_null, 1, sd_null, 1, 0)

AIC = -2 * LL + 2 * 5
BIC = -2 * LL + 5 * np.log(len(ADoptimal))

AIC_null = -2 * LL_null + 2 * 2
BIC_null = -2 * LL_null + 2 * np.log(len(ADoptimal))

R2 = 1 - (LL / LL_null)





