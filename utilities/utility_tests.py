import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def correlation_test(df, trial, condition, variable_of_interest, checker=False, sig_only=False):
    # Preallocate a list to store the results
    results = []

    # separate by trial and condition
    df = df[(df['ChoiceSet'] == trial) & (df['Condition'] == condition)]

    # Loop through the personality scales
    for scale in df.columns[4:20]:
        r, p = stats.pearsonr(df[scale], df[variable_of_interest])
        results.append({'personality_scale': scale, 'r': r, 'p': p})

    if checker:
        # Check if there IS a significant correlation
        if any(result['p'] < 0.05 for result in results):
            print(
                f"[{variable_of_interest}]: There are significant correlations for trial {trial} in the {condition} condition")
        else:
            # state that there is no significant correlation
            print(
                f"[{variable_of_interest}]: There are no significant correlations for trial {trial} in the {condition} condition")

    else:
        # Convert the list to a DataFrame
        results_df = pd.DataFrame(results, columns=['personality_scale', 'r', 'p'])
        # Check if there IS a significant correlation
        if any(result['p'] < 0.05 for result in results):
            if sig_only:
                # return only the significant correlations
                return results_df[results_df['p'] < 0.05]
            else:
                # return all the correlations
                return results_df
        else:
            print("No significant correlations")
            return results_df


def normality_test(trial_list, method, variable, condition_list=None, distribution='norm'):
    # preallocate a list to store the results
    results = []

    # loop through the trials
    if method == 'anderson':
        if condition_list is None:
            for trial in trial_list:
                result = stats.anderson(trial[variable], dist=distribution)
                results.append({'trial': trial['ChoiceSet'].iloc[0], 'statistic': result.statistic,
                                'critical_values': result.critical_values,
                                'significance_level': result.significance_level})
            results = pd.DataFrame(results,
                                   columns=['trial', 'statistic', 'critical_values', 'significance_level'])
        else:
            for trial in trial_list:
                for condition in condition_list:
                    result = stats.anderson(trial[trial['Condition'] == condition][variable], dist=distribution)
                    results.append({'trial': trial['ChoiceSet'].iloc[0], 'condition': condition,
                                    'statistic': result.statistic, 'critical_values': result.critical_values,
                                    'significance_level': result.significance_level})
            results = pd.DataFrame(results,
                                   columns=['trial', 'condition', 'statistic', 'critical_values',
                                            'significance_level'])
    elif method == 'shapiro':
        if condition_list is None:
            for trial in trial_list:
                w, p = stats.shapiro(trial[variable])
                results.append({'trial': trial['ChoiceSet'].iloc[0], 'statistic': w,
                                'significance_level': p})
            results = pd.DataFrame(results,
                                   columns=['trial', 'statistic', 'significance_level'])
        else:
            for trial in trial_list:
                for condition in condition_list:
                    w, p = stats.shapiro(trial[trial['Condition'] == condition][variable])
                    results.append({'trial': trial['ChoiceSet'].iloc[0], 'condition': condition,
                                    'statistic': w, 'significance_level': p})
            results = pd.DataFrame(results,
                                   columns=['trial', 'condition', 'statistic', 'significance_level'])
    else:
        raise ValueError('Invalid method')

    return results


def perform_gmm(data, n_components):
    # fit the model
    gmm = GaussianMixture(n_components=n_components, means_init=[[0], [1]], max_iter=10000, random_state=0)
    gmm.fit(data)

    # Getting the fitted parameters
    weights = gmm.weights_
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()

    # Getting the model fit
    aic = gmm.aic(data)
    bic = gmm.bic(data)

    # # Plotting the original data as a histogram
    # plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    #
    # # Plotting the fitted bimodal distribution
    # xmin, xmax = plt.xlim()
    # x = np.linspace(xmin, xmax, 100)
    # for i in range(2):
    #     plt.plot(x, weights[i] * stats.norm.pdf(x, means[i], np.sqrt(covariances[i])), color='r')
    #
    # plt.title('Fitting a Bimodal Distribution')
    # plt.xlabel('Data')
    # plt.ylabel('Proportion of Optimal Choices')
    # plt.show()

    return weights, means, covariances, aic, bic


def bimodal_test(trial_list, condition_list, n_components=2):
    # preallocate a list to store the results
    results = []

    if condition_list is None:
        # loop through the trials
        for trial in trial_list:
            data = np.array(trial['PropOptimal']).reshape(-1, 1)
            weights, means, covariances, aic, bic = perform_gmm(data, n_components)

            # append the results
            results.append({'trial': trial['ChoiceSet'].iloc[0], 'weights_1': weights[0], 'weights_2': weights[1],
                            'means_1': means[0], 'means_2': means[1], 'covariances_1': covariances[0],
                            'covariances_2': covariances[1], 'aic': aic, 'bic': bic})
            # Convert the list to a DataFrame
            results_df = pd.DataFrame(results, columns=['trial', 'weights_1', 'weights_2', 'means_1', 'means_2',
                                                        'covariances_1', 'covariances_2', 'aic', 'bic'])
    else:
        for trial in trial_list:
            for condition in condition_list:
                data = np.array(trial[trial['Condition'] == condition]['PropOptimal']).reshape(-1, 1)
                weights, means, covariances, aic, bic = perform_gmm(data, n_components)

                # append the results
                results.append({'trial': trial['ChoiceSet'].iloc[0], 'condition': condition, 'weights_1': weights[0],
                                'weights_2': weights[1], 'means_1': means[0], 'means_2': means[1],
                                'covariances_1': covariances[0], 'covariances_2': covariances[1], 'aic': aic,
                                'bic': bic})

        # Convert the list to a DataFrame
        results_df = pd.DataFrame(results,
                                  columns=['trial', 'condition', 'weights_1', 'weights_2', 'means_1', 'means_2',
                                           'covariances_1', 'covariances_2', 'aic', 'bic'])

    return results_df
