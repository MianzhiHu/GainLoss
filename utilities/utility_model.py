import numpy as np
import pandas as pd
from scipy.stats import norm


def log_likelihood(dat, mu1, mu2, sd1, sd2, ppi):
    ll = np.sum(np.log((1 - ppi) * norm.pdf(dat, mu1, sd1) + ppi * norm.pdf(dat, mu2, sd2)))
    return ll


def e_step(dat, mu1, mu2, sd1, sd2, ppi):
    # Calculate the responsibility using the current parameter estimates
    resp = ppi * norm.pdf(dat, mu2, sd2) / ((1 - ppi) * norm.pdf(dat, mu1, sd1) + ppi * norm.pdf(dat, mu2, sd2))
    return resp


def m_step(dat, resp):
    # Update the parameters using the current responsibilities
    mu1 = np.sum((1 - resp) * dat) / np.sum(1 - resp)
    mu2 = np.sum(resp * dat) / np.sum(resp)
    sd1 = np.sqrt(np.sum((1 - resp) * (dat - mu1) ** 2) / np.sum(1 - resp))
    sd2 = np.sqrt(np.sum(resp * (dat - mu2) ** 2) / np.sum(resp))
    ppi = np.mean(resp)
    return mu1, mu2, sd1, sd2, ppi


def initialize_parameters(data):
    # Bounded mu initialization
    mu = np.random.uniform(data.min(), data.max())

    # Bounded sd initialization
    sd = np.random.uniform(0.01, (data.max() - data.min()) / 2.0)

    return mu, sd


def em_model(data, n_iter=1000, tolerance=0.0001, random_init=True):

    if random_init:
        # Starting parameter estimates
        mu1, sd1 = initialize_parameters(data)
        mu2, sd2 = initialize_parameters(data)
        ppi = np.random.uniform(0, 1)

    else:
        # Starting parameter estimates
        mu1, sd1 = 0, np.std(data)
        mu2, sd2 = 1, np.std(data)
        ppi = 0.5

    # Assuming your data is stored in a list or numpy array named dat
    change = np.inf
    oldppi = 0

    for i in range(n_iter):
        if change > tolerance:
            # E-Step
            resp = e_step(data, mu1, mu2, sd1, sd2, ppi)
            # M-Step
            mu1, mu2, sd1, sd2, newppi = m_step(data, resp)

            change = np.abs(newppi - oldppi)
            oldppi = newppi
            ppi = newppi
        else:
            print("Iteration stopped at iteration: ", i)
            break

    # Calculate the log likelihood for each observation
    ll = log_likelihood(data, mu1, mu2, sd1, sd2, ppi)

    # Calculate the AIC and BIC
    n = len(data)
    k = 5
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)

    # compare with a single normal distribution
    mu, sd = norm.fit(data)
    ll_null = log_likelihood(data, mu, sd, 1, 1, 0)
    aic_null = -2 * ll + 2 * 2
    bic_null = -2 * ll + 2 * np.log(n)

    # calculate the R2
    R2 = 1 - ll / ll_null

    return mu1, mu2, sd1, sd2, ppi, ll, ll_null, aic, aic_null, bic, bic_null, R2

