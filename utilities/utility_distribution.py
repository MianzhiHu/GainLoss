import numpy as np
import pandas as pd
from scipy.stats import norm


def log_likelihood(dat, mu1, mu2, sd1, sd2, ppi1, modality='bimodal',
                   mu3=None, sd3=None, ppi2=None, ppi3=None):
    if modality == 'trimodal':
        ll = np.sum(np.log((ppi1) * norm.pdf(dat, mu1, sd1) + ppi2 * norm.pdf(dat, mu2, sd2) +
                             ppi3 * norm.pdf(dat, mu3, sd3)))

        return ll

    else:
        ll = np.sum(np.log((1 - ppi1) * norm.pdf(dat, mu1, sd1) + ppi1 * norm.pdf(dat, mu2, sd2)))

        return ll


def e_step(dat, mu1, mu2, sd1, sd2, ppi1, modality='bimodal', mu3=None, sd3=None, ppi2=None, ppi3=None):
    if modality == 'trimodal':
        # get likelihood of each observation under each component
        p1 = ppi1 * norm.pdf(dat, mu1, sd1)
        p2 = ppi2 * norm.pdf(dat, mu2, sd2)
        p3 = ppi3 * norm.pdf(dat, mu3, sd3)

        # calculate the responsibility using the current parameter estimates
        resp1 = p1 / (p1 + p2 + p3)
        resp2 = p2 / (p1 + p2 + p3)
        resp3 = p3 / (p1 + p2 + p3)

        return resp1, resp2, resp3

    else:
        # Calculate the responsibility using the current parameter estimates
        resp = ppi1 * norm.pdf(dat, mu2, sd2) / ((1 - ppi1) * norm.pdf(dat, mu1, sd1) + ppi1 * norm.pdf(dat, mu2, sd2))
        return resp


def m_step(dat, resp1, modality='bimodal', resp2=None, resp3=None):
    if modality == 'trimodal':
        ppi1 = np.mean(resp1)
        ppi2 = np.mean(resp2)
        ppi3 = np.mean(resp3)

        mu1 = np.sum(resp1 * dat) / np.sum(resp1)
        mu2 = np.sum(resp2 * dat) / np.sum(resp2)
        mu3 = np.sum(resp3 * dat) / np.sum(resp3)

        sd1 = np.sqrt(np.sum(resp1 * (dat - mu1) ** 2) / np.sum(resp1))
        sd2 = np.sqrt(np.sum(resp2 * (dat - mu2) ** 2) / np.sum(resp2))
        sd3 = np.sqrt(np.sum(resp3 * (dat - mu3) ** 2) / np.sum(resp3))

        return mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3

    else:
        # Update the parameters using the current responsibilities
        mu1 = np.sum((1 - resp1) * dat) / np.sum(1 - resp1)
        mu2 = np.sum(resp1 * dat) / np.sum(resp1)
        sd1 = np.sqrt(np.sum((1 - resp1) * (dat - mu1) ** 2) / np.sum(1 - resp1))
        sd2 = np.sqrt(np.sum(resp1 * (dat - mu2) ** 2) / np.sum(resp1))
        ppi = np.mean(resp1)
        return mu1, mu2, sd1, sd2, ppi


def initialize_parameters(data, size):
    # Bounded mu initialization
    mu = np.random.uniform(data.min(), data.max(), size)

    # Bounded sd initialization
    sd = np.random.uniform(0.01, (data.max() - data.min()) / 2.0, size)

    return mu, sd


def em_model(data, tolerance=0.0001, random_init=True, return_starting_params=False, modality='bimodal',
             mu1=None, mu2=None, mu3=None, sd1=None, sd2=None, sd3=None, ppi1=None, ppi2=None):

    # set global variables
    change = np.inf

    if modality == 'trimodal':
        if random_init:
            # randomly generate starting mu and sd
            mu1, sd1 = initialize_parameters(data, 1)
            mu2, sd2 = initialize_parameters(data, 1)
            mu3, sd3 = initialize_parameters(data, 1)

            # randomly generate a starting ppi
            ppi1 = np.random.uniform(0.01, 1)
            ppi2 = np.random.uniform(0.01, 1-ppi1)
            ppi3 = 1 - ppi1 - ppi2

            # record the starting parameters
            (starting_mu1, starting_mu2, starting_mu3, starting_sd1, starting_sd2,
             starting_sd3, starting_ppi1, starting_ppi2, starting_ppi3) = (mu1[0], mu2[0], mu3[0],
                                                                          sd1[0], sd2[0], sd3[0],
                                                                            ppi1, ppi2, ppi3)

        else:
            # Starting parameter estimates
            mu1, sd1 = mu1, sd1
            mu2, sd2 = mu2, sd2
            mu3, sd3 = mu3, sd3
            ppi1 = ppi1
            ppi2 = ppi2
            ppi3 = 1 - ppi1 - ppi2

        # Assuming your data is stored in a list or numpy array named dat
        oldppi1 = 0
        oldppi2 = 0
        oldppi3 = 0

        while change > tolerance:

            # E-Step
            resp1, resp2, resp3 = e_step(data, mu1, mu2, sd1, sd2, ppi1, modality='trimodal', mu3=mu3, sd3=sd3,
                                            ppi2=ppi2, ppi3=ppi3)
            # M-Step
            mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3 = m_step(data, resp1, modality='trimodal', resp2=resp2,
                                                                      resp3=resp3)

            change1 = np.abs(ppi1 - oldppi1)
            change2 = np.abs(ppi2 - oldppi2)
            change3 = np.abs(ppi3 - oldppi3)

            change = max(change1, change2, change3)

            oldppi1 = ppi1
            oldppi2 = ppi2
            oldppi3 = ppi3

        # make sure the larger mean is always mu1
        if mu1 < mu2:
            mu1, mu2 = mu2, mu1
            sd1, sd2 = sd2, sd1
            ppi1, ppi2 = ppi2, ppi1

        if mu1 < mu3:
            mu1, mu3 = mu3, mu1
            sd1, sd3 = sd3, sd1
            ppi1, ppi3 = ppi3, ppi1

        if mu2 < mu3:
            mu2, mu3 = mu3, mu2
            sd2, sd3 = sd3, sd2
            ppi2, ppi3 = ppi3, ppi2

        # Calculate the log likelihood for each observation
        ll = log_likelihood(data, mu1, mu2, sd1, sd2, ppi1, modality='trimodal',
                            mu3=mu3, sd3=sd3, ppi2=ppi2, ppi3=ppi3)

    else:
        if random_init:
            # randomly generate starting mu and sd
            mu1, sd1 = initialize_parameters(data, 1)
            mu2, sd2 = initialize_parameters(data, 1)

            # randomly generate a starting ppi
            ppi = np.random.uniform(0.01, 1)

            # record the starting parameters
            starting_mu1, starting_mu2, starting_sd1, starting_sd2, starting_ppi = (mu1[0], mu2[0],
                                                                                    sd1[0], sd2[0], ppi)

        else:
            # Starting parameter estimates
            mu1, sd1 = mu1, sd1
            mu2, sd2 = mu2, sd2
            ppi = ppi1

        # Assuming your data is stored in a list or numpy array named dat
        oldppi = 0

        while change > tolerance:

            # E-Step
            resp1 = e_step(data, mu1, mu2, sd1, sd2, ppi)
            # M-Step
            mu1, mu2, sd1, sd2, ppi = m_step(data, resp1)

            change = np.abs(ppi - oldppi)

            oldppi = ppi

        # make sure the larger mean is always mu1
        if mu1 < mu2:
            mu1, mu2 = mu2, mu1
            sd1, sd2 = sd2, sd1
            ppi = 1 - ppi

        # Calculate the log likelihood for each observation
        ll = log_likelihood(data, mu1, mu2, sd1, sd2, ppi)

    # Calculate the AIC and BIC
    if modality == 'trimodal':
        k = 8
    else:
        k = 5

    n = len(data)
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)

    # compare with a single normal distribution
    mu_null, sd_null = norm.fit(data)
    ll_null = log_likelihood(data, mu_null, 1, sd_null, 1, 0)
    aic_null = -2 * ll_null + 2 * 2
    bic_null = -2 * ll_null + 2 * np.log(n)

    # calculate the R2
    R2 = 1 - ll / ll_null

    if modality == 'trimodal':
        if return_starting_params:
            return (starting_mu1, starting_mu2, starting_mu3, starting_sd1, starting_sd2,
                    starting_sd3, starting_ppi1, starting_ppi2, starting_ppi3,
                    mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3, ll, ll_null, aic, aic_null, bic, bic_null, R2)

        else:
            return mu1, mu2, mu3, sd1, sd2, sd3, ppi1, ppi2, ppi3, ll, ll_null, aic, aic_null, bic, bic_null, R2

    else:
        if return_starting_params:
            return (starting_mu1, starting_mu2, starting_sd1, starting_sd2, starting_ppi,
                    mu1, mu2, sd1, sd2, ppi, ll, ll_null, aic, aic_null, bic, bic_null, R2)

        else:
            return mu1, mu2, sd1, sd2, ppi, ll, ll_null, aic, aic_null, bic, bic_null, R2