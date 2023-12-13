import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm, beta
from utilities.utility_distribution import (em_model, pdf_plot_generator, likelihood_ratio_test,
                                            parameter_extractor, group_assignment)
import matplotlib.pyplot as plt
from utilities.utility_tests import normality_test, bimodal_test

# Read in the data
data = pd.read_csv('./data/data.csv')
data = data[data['Condition'].isin(['Gains', 'Losses'])]
E1 = pd.read_csv('./data/PropOptimal_E1.csv')
E2 = pd.read_csv('./data/PropOptimal_E2.csv')
E3 = pd.read_csv('./data/PropOptimal_E3.csv')

# Read in the results if necessary
trimodal_CA = pd.read_csv('./data/trimodal_CA_fre.csv')
trimodal_BD = pd.read_csv('./data/trimodal_BD.csv')
bimodal_CA = pd.read_csv('./data/bimodal_CA.csv')

# bimodal_CA = bimodal_CA[bimodal_CA['mu1'] < 0.9]

# combine E1 and E2 so that we have enough data points
E_F = pd.concat([E1, E2], ignore_index=True)
E_F_CA = E_F[E_F['ChoiceSet'] == 'CA']['PropOptimal']
E3_CA = E3[E3['ChoiceSet'] == 'CA']['PropOptimal']
E_F_all = pd.concat([E1, E2, E3], ignore_index=True)
E_F_all_CA = E_F_all[E_F_all['ChoiceSet'] == 'CA']['PropOptimal']

# Subset the data
ABoptimal = data[data['ChoiceSet'] == 'AB']['PropOptimal']
CDoptimal = data[data['ChoiceSet'] == 'CD']['PropOptimal']
CAoptimal = data[data['ChoiceSet'] == 'CA']['PropOptimal']
BDoptimal = data[data['ChoiceSet'] == 'BD']['PropOptimal']
CBoptimal = data[data['ChoiceSet'] == 'CB']['PropOptimal']
ADoptimal = data[data['ChoiceSet'] == 'AD']['PropOptimal']

CAoptimal_Gains = data[(data['ChoiceSet'] == 'CA') & (data['Condition'] == 'Gains')]['PropOptimal']


# validate the model with 1000 iterations
# In BD trials, too many participants had 0 optimal choices, which makes the model fail to converge
n_iter = 1000

# result = []
# for i in range(n_iter):
#     print(i)
#     mu1, mu2, sd1, sd2, ppi, ll, ll_null, aic, aic_null, bic, bic_null, R2\
#         = em_model(CAoptimal, tolerance=1e-10, random_init=True)
#     result.append([mu1, mu2, sd1, sd2, ppi, ll, ll_null, aic, aic_null, bic, bic_null, R2])
#
# # convert the result to a dataframe
# result = pd.DataFrame(result, columns=['mu1', 'mu2', 'sd1', 'sd2', 'ppi', 'll', 'll_null',
#                                        'aic', 'aic_null', 'bic', 'bic_null', 'R2'])
# result = result.dropna()
# # round the result to 3 decimal places
# result = result.round(3)
# # result.to_csv('./data/bimodal_CA.csv', index=False)
#
# print(result['mu1'].value_counts())
#
# pdf_plot_generator(CAoptimal, bimodal_CA, 'bimodal')
# # print(trimodal_CA['mu1'].value_counts())

# only run if needed because it takes a long time with 10000 iterations
# let's see how the model converge with 3 components
result_tri = []
for i in range(n_iter):
    print(i)
    mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3, ll, ll_null, aic, aic_null, bic, bic_null, R2\
        = em_model(CAoptimal, tolerance=1e-10, random_init=True, modality='trimodal')

    result_tri.append([mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3, ll, ll_null, aic, aic_null, bic, bic_null, R2])

# convert the result to a dataframe
result_tri = pd.DataFrame(result_tri, columns=['mu1', 'mu2', 'mu3', 'sd1', 'sd2', 'sd3', 'ppi1', 'ppi2', 'ppi3', 'll',
                                             'll_null', 'aic', 'aic_null', 'bic', 'bic_null', 'R2'])

# round the result to 3 decimal places
result_tri = result_tri.dropna()
result_tri = result_tri.round(3)
result_tri = result_tri.applymap(lambda x: "{:.3f}".format(x) if isinstance(x, (int, float)) else x)

print(result_tri['mu1'].value_counts())

pdf_plot_generator(CAoptimal, result_tri, 'trimodal')

likelihood_ratio_test(trimodal_CA, 6)

# save the result to a csv file
result_tri.to_csv('./data/trimodal_CA_fre.csv', index=False)

pdf_plot_generator(CAoptimal, trimodal_CA, 'trimodal')

# now assign participants to each group
assignments_CA = group_assignment(CAoptimal, trimodal_CA, 'trimodal')
# assignments_BD = group_assignment(BDoptimal, trimodal_BD, 'trimodal')
# assignments_CA.to_csv('./data/trimodal_assignments_CA.csv', index=False)

# print(trimodal_CA['mu1'].value_counts())