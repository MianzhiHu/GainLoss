import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm, beta
from utils.Expectation_Maximization import (em_model, pdf_plot_generator, likelihood_ratio_test,
                                            parameter_extractor, group_assignment)
import matplotlib.pyplot as plt
from utilities.utility_tests import normality_test, bimodal_test

# Read in the data
E1_PropOptimal = pd.read_csv('./data/PropOptimal_E1.csv')
E2_PropOptimal = pd.read_csv('./data/propoptimal_id.csv')
E1_data = pd.read_csv('./data/data_gains.csv')
E2_data = pd.read_csv('./data/data_id.csv')

# Subset the data
E1_CA_Baseline = E1_PropOptimal[(E1_PropOptimal['Condition'] == 'GainsEF') & (E1_PropOptimal['SetSeen.'] == 2)]
E1_CA_Frequency = E1_PropOptimal[(E1_PropOptimal['Condition'] == 'Gains') & (E1_PropOptimal['SetSeen.'] == 2)]
E2_CA_Baseline = E2_PropOptimal[(E2_PropOptimal['Condition'] == 'Baseline') & (E2_PropOptimal['TrialType'] == 'CA')]
E2_CA_Frequency = E2_PropOptimal[(E2_PropOptimal['Condition'] == 'Frequency') & (E2_PropOptimal['TrialType'] == 'CA')]

# validate the model with 1000 iterations
# In BD trials, too many participants had 0 optimal choices, which makes the model fail to converge
n_iter = 10000
df_of_interest = E2_CA_Frequency

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

# # only run if needed because it takes a long time with 10000 iterations
# # let's see how the model converge with 3 components
# result_tri = []
# for i in range(n_iter):
#     print(i)
#     mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3, ll, ll_null, aic, aic_null, bic, bic_null, R2\
#         = em_model(df_of_interest['BestOption'].values, tolerance=1e-12, random_init=True, modality='trimodal')
#
#     result_tri.append([mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3, ll, ll_null, aic, aic_null, bic, bic_null, R2])
#
# # convert the result to a dataframe
# result_tri = pd.DataFrame(result_tri, columns=['mu1', 'mu2', 'mu3', 'sd1', 'sd2', 'sd3', 'ppi1', 'ppi2', 'ppi3', 'll',
#                                              'll_null', 'aic', 'aic_null', 'bic', 'bic_null', 'R2'])
#
# # round the result to 3 decimal places
# result_tri = result_tri.dropna()
# result_tri = result_tri.round(3)
# result_tri = result_tri.map(lambda x: "{:.3f}".format(x) if isinstance(x, (int, float)) else x)
#
# print(result_tri['mu1'].value_counts())
#

# # likelihood_ratio_test(result_tri, 6)
#
# # save the result to a csv file
# result_tri.to_csv('./data/E2_Fre_trimodal.csv', index=False)

# ======================================================================================================================
# Add the results into the data
# ======================================================================================================================
# Read in the trimodal assignments
E1_Bas_trimodal = pd.read_csv('./data/E1_Bas_trimodal.csv')
E1_Fre_trimodal = pd.read_csv('./data/E1_Fre_trimodal.csv')
E2_Bas_trimodal = pd.read_csv('./data/E2_Bas_trimodal.csv')
E2_Fre_trimodal = pd.read_csv('./data/E2_Fre_trimodal.csv')

# now assign participants to each group
E1_Bas_tri_assignments = group_assignment(E1_CA_Baseline['BestOption'].values, E1_Bas_trimodal, 'trimodal')
E1_Fre_tri_assignments = group_assignment(E1_CA_Frequency['BestOption'].values, E1_Fre_trimodal, 'trimodal')
E2_Bas_tri_assignments = group_assignment(E2_CA_Baseline['BestOption'].values, E2_Bas_trimodal, 'trimodal')
E2_Fre_tri_assignments = group_assignment(E2_CA_Frequency['BestOption'].values, E2_Fre_trimodal, 'trimodal')

# Add participant number and rename the columns
E1_Bas_tri_assignments['Subnum'] = E1_CA_Baseline['Subnum'].values
E1_Fre_tri_assignments['Subnum'] = E1_CA_Frequency['Subnum'].values
E2_Bas_tri_assignments['Subnum'] = E2_CA_Baseline['Subnum'].values
E2_Fre_tri_assignments['Subnum'] = E2_CA_Frequency['Subnum'].values

# Rename the columns
E2_Bas_tri_assignments.rename(columns={'prob1': 'prob1_baseline', 'prob2': 'prob2_baseline', 'prob3': 'prob3_baseline',
                              'assignments': 'group_baseline'}, inplace=True)
E2_Fre_tri_assignments.rename(columns={'prob1': 'prob1_frequency', 'prob2': 'prob2_frequency', 'prob3': 'prob3_frequency',
                                'assignments': 'group_frequency'}, inplace=True)

# Merge the assignments with the original data
E1_group_assignments = pd.concat([E1_Bas_tri_assignments, E1_Fre_tri_assignments], axis=0)
E1_data = pd.merge(E1_data, E1_group_assignments, on='Subnum', how='left')
E1_PropOptimal = pd.merge(E1_PropOptimal, E1_group_assignments, on='Subnum', how='left')

E2_group_assignments = pd.merge(E2_Bas_tri_assignments, E2_Fre_tri_assignments, on='Subnum', how='left')
E2_data = pd.merge(E2_data, E2_group_assignments, on='Subnum', how='left')
E2_PropOptimal = pd.merge(E2_PropOptimal, E2_group_assignments, on='Subnum', how='left')

# Plot the results
pdf_plot_generator(E1_CA_Baseline['BestOption'].values, E1_Bas_trimodal, './figures/trimodal_E1_Baseline.png', modality='trimodal', x_label='% of C Choices', y_label='Probability Density', density=True)
pdf_plot_generator(E1_CA_Frequency['BestOption'].values, E1_Fre_trimodal, './figures/trimodal_E1_Frequency.png', modality='trimodal', x_label="% of C Choices", y_label='Probability Density', density=True)
pdf_plot_generator(E2_CA_Baseline['BestOption'].values, E2_Bas_trimodal, './figures/trimodal_E2_Baseline.png', modality='trimodal', x_label="% of C Choices", y_label='Probability Density', density=True)
pdf_plot_generator(E2_CA_Frequency['BestOption'].values, E2_Fre_trimodal, './figures/trimodal_E2_Frequency.png', modality='trimodal', x_label="% of C Choices", y_label='Probability Density', density=True)


# Save the updated data
E1_data.to_csv('./data/E1_data_with_assignments.csv', index=False)
E1_PropOptimal.to_csv('./data/E1_summary_with_assignments.csv', index=False)
E2_data.to_csv('./data/E2_data_with_assignments.csv', index=False)
E2_PropOptimal.to_csv('./data/E2_summary_with_assignments.csv', index=False)
