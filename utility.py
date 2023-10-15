import pandas as pd
import scipy.stats as stats


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
