import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from utilities.utility_distribution import em_model, pdf_plot_generator
import matplotlib.pyplot as plt
from utilities.utility_tests import normality_test, bimodal_test

# Read in the data
data = pd.read_csv('./data/data.csv')
E1 = pd.read_csv('./data/PropOptimal_E1.csv')
E2 = pd.read_csv('./data/PropOptimal_E2.csv')
E3 = pd.read_csv('./data/PropOptimal_E3.csv')

# remove the outlier trials (RT > 10s)
# other files don't have this problem
data = data[data['RT'] < 10000]

data.loc[data['PropOptimal'] == 1, 'PropOptimal'] = 0.9999
data.loc[data['PropOptimal'] == 0, 'PropOptimal'] = 0.0001
# Subset the data
ABoptimal = data[data['ChoiceSet'] == 'AB']['PropOptimal']
CDoptimal = data[data['ChoiceSet'] == 'CD']['PropOptimal']
CAoptimal = data[data['ChoiceSet'] == 'CA']['PropOptimal']
BDoptimal = data[data['ChoiceSet'] == 'BD']['PropOptimal']
CBoptimal = data[data['ChoiceSet'] == 'CB']['PropOptimal']
ADoptimal = data[data['ChoiceSet'] == 'AD']['PropOptimal']


# validate the model with 1000 iterations
n_iter = 5
result = []
for i in range(n_iter):
    print(i)
    (starting_mu1, starting_mu2, starting_sd1, starting_sd2, starting_ppi, mu1, mu2, sd1, sd2, ppi, ll, ll_null, aic,
     aic_null, bic, bic_null, R2)\
        = em_model(CAoptimal, tolerance=1e-10, random_init=True, return_starting_params=True)
    result.append([starting_mu1, starting_mu2, starting_sd1, starting_sd2, starting_ppi, mu1, mu2, sd1, sd2, ppi, ll,
                   ll_null, aic, aic_null, bic, bic_null, R2])

# convert the result to a dataframe
result = pd.DataFrame(result, columns=['starting_mu1', 'starting_mu2', 'starting_sd1', 'starting_sd2', 'starting_ppi',
                                        'mu1', 'mu2', 'sd1', 'sd2', 'ppi', 'll', 'll_null', 'aic', 'aic_null', 'bic',
                                        'bic_null', 'R2'])
result = result.dropna()
# round the result to 3 decimal places
result = result.round(3)

print(result['mu1'].value_counts())

pdf_plot_generator(CAoptimal, result, 'bimodal')

# let's see how the model converge with 3 components
result_3 = []
for i in range(n_iter):
    print(i)
    mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3, ll, ll_null, aic, aic_null, bic, bic_null, R2\
        = em_model(CAoptimal, tolerance=1e-10, random_init=True, modality='trimodal')

    result_3.append([mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3, ll, ll_null, aic, aic_null, bic, bic_null, R2])

# convert the result to a dataframe
result_3 = pd.DataFrame(result_3, columns=['mu1', 'mu2', 'mu3', 'sd1', 'sd2', 'sd3', 'ppi1', 'ppi2', 'ppi3', 'll',
                                             'll_null', 'aic', 'aic_null', 'bic', 'bic_null', 'R2'])

# round the result to 3 decimal places
result_3 = result_3.dropna()
result_3 = result_3.round(3)

pdf_plot_generator(CAoptimal, result_3, 'trimodal')
