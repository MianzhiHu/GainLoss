import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from utilities.utility_distribution import em_model, e_step, m_step, log_likelihood, initialize_parameters
import matplotlib.pyplot as plt
from utilities.utility_tests import normality_test, bimodal_test

# Read in the data
data = pd.read_csv('./data/data.csv')
E1 = pd.read_csv('./data/PropOptimal_E1.csv')

# trial_list = [1, 2, 3, 4, 5, 6]
#
# ABoptimal = E1[E1['optSeen'] == 1]
# CDoptimal = E1[E1['optSeen'] == 2]
# CAoptimal = E1[E1['optSeen'] == 3]
# BDoptimal = E1[E1['optSeen'] == 4]
# CBoptimal = E1[E1['optSeen'] == 5]
# ADoptimal = E1[E1['optSeen'] == 6]
#
# trial_list = [ABoptimal, CDoptimal, CAoptimal, BDoptimal, CBoptimal, ADoptimal]
# condition_list = ['Losses', 'LossesEF', 'Gains', 'GainsEF']
#
# normality_E1 = normality_test(trial_list, 'shapiro', 'PropOptimal', condition_list=None, distribution='norm')
#



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

CAoptimal_E1 = E1[E1['optSeen'] == 3]['PropOptimal']

# now fit the em model to each of the six choice sets

mu1, mu2, sd1, sd2, ppi, ll, ll_null, aic, aic_null, bic, bic_null, R2\
    = em_model(CAoptimal, tolerance=1e-10, random_init=False,
                mu1=1, mu2=0, sd1=0.2, sd2=0.2, ppi1=0.5)

# validate the model with 1000 iterations
n_iter = 100
result = []
for i in range(n_iter):
    print(i)
    (starting_mu1, starting_mu2, starting_sd1, starting_sd2, starting_ppi, mu1, mu2, sd1, sd2, ppi, ll, ll_null, aic,
     aic_null, bic, bic_null, R2)\
        = em_model(CAoptimal_E1, tolerance=1e-10, random_init=True, return_starting_params=True)
    result.append([starting_mu1, starting_mu2, starting_sd1, starting_sd2, starting_ppi, mu1, mu2, sd1, sd2, ppi, ll,
                   ll_null, aic, aic_null, bic, bic_null, R2])

# convert the result to a dataframe
result = pd.DataFrame(result, columns=['starting_mu1', 'starting_mu2', 'starting_sd1', 'starting_sd2', 'starting_ppi',
                                        'mu1', 'mu2', 'sd1', 'sd2', 'ppi', 'll', 'll_null', 'aic', 'aic_null', 'bic',
                                        'bic_null', 'R2'])
result = result.dropna()
# round the result to 3 decimal places
result = result.round(3)

# make everything as a numeric variable
result = result.astype({'starting_mu1': 'float', 'starting_mu2': 'float', 'starting_sd1': 'float', 'starting_sd2': 'float',
                        'starting_ppi': 'float'})
print(result['mu1'].value_counts())

# summarize those simulations that converge to the correct parameters
result_converge = result[result['mu1'] == 0.952]
print(result_converge['starting_mu2'].describe())


result_not_converge = result[result['mu1'] == 0.572]
print(result_not_converge['starting_mu1'].describe())


plt.hist(CAoptimal, bins=50, density=True, alpha=0.6, color='g', label="Data")
x = np.linspace(min(CAoptimal), max(CAoptimal), 1000)

# Calculate the Gaussian distributions
pdf1 = norm.pdf(x, .572, .278)
pdf2 = norm.pdf(x, .041, .037)

# Weight the pdfs by the mixing coefficients
weighted_pdf1 = (1-.223) * pdf1
weighted_pdf2 = .223 * pdf2

# Plot
plt.plot(x, weighted_pdf1, 'k', linewidth=2, label=f"Component 1: $\mu$={mu1:.2f}, $\sigma$={sd1:.2f}")
plt.plot(x, weighted_pdf2, 'r', linewidth=2, label=f"Component 2: $\mu$={mu2:.2f}, $\sigma$={sd2:.2f}")

plt.title("EM Fitted Gaussian Mixture Model")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()


plt.hist(CAoptimal, bins=50, density=True, alpha=0.6, color='g', label="Data")
x = np.linspace(min(CAoptimal), max(CAoptimal), 1000)

# Calculate the Gaussian distributions
pdf1 = norm.pdf(x, .952, .04)
pdf2 = norm.pdf(x, .355, .268)

# Weight the pdfs by the mixing coefficients
weighted_pdf1 = (1-.835) * pdf1
weighted_pdf2 = .835 * pdf2

# Plot
plt.plot(x, weighted_pdf1, 'k', linewidth=2, label=f"Component 1: $\mu$={mu1:.2f}, $\sigma$={sd1:.2f}")
plt.plot(x, weighted_pdf2, 'r', linewidth=2, label=f"Component 2: $\mu$={mu2:.2f}, $\sigma$={sd2:.2f}")

plt.title("EM Fitted Gaussian Mixture Model")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

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

# make everything as a numeric variable
result_3 = result_3.astype({'mu1': 'float', 'mu2': 'float', 'mu3': 'float', 'sd1': 'float', 'sd2': 'float', 'sd3': 'float',
                            'ppi1': 'float', 'ppi2': 'float', 'ppi3': 'float'})

# visualize the result
plt.hist(CAoptimal_E1, bins=50, density=True, alpha=0.6, color='g', label="Data")
x = np.linspace(min(CAoptimal_E1), max(CAoptimal_E1), 1000)

# Calculate the Gaussian distributions
pdf1 = norm.pdf(x, result_3['mu1'].mode(), result_3['sd1'].mode())
pdf2 = norm.pdf(x, result_3['mu2'].mode(), result_3['sd2'].mode())
pdf3 = norm.pdf(x, result_3['mu3'].mode(), result_3['sd3'].mode())

# Weight the pdfs by the mixing coefficients
weighted_pdf1 = .132 * pdf1
weighted_pdf2 = .741 * pdf2
weighted_pdf3 = .127 * pdf3

# Plot
plt.plot(x, weighted_pdf1, 'k', linewidth=2, label=f"Component 1: $\mu$={mu1:.2f}, $\sigma$={sd1:.2f}")
plt.plot(x, weighted_pdf2, 'r', linewidth=2, label=f"Component 2: $\mu$={mu2:.2f}, $\sigma$={sd2:.2f}")
plt.plot(x, weighted_pdf3, 'b', linewidth=2, label=f"Component 3: $\mu$={mu3:.2f}, $\sigma$={sd3:.2f}")

plt.title("EM Fitted Gaussian Mixture Model")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

beta.fit(CAoptimal)
