import numpy as np
import pandas as pd
import itertools
import functools
from utils.ComputationalModeling import (ComputationalModels, dict_generator, parameter_extractor, weight_calculation,
                                         bayes_factor, get_best_df, vb_model_selection, compute_exceedance_prob)

# Read in the data
data = pd.read_csv('data.csv')

# Some data cleaning
data = data.reset_index(drop=True)
data.rename(columns={'TrialType': 'SetSeen '}, inplace=True)
mapping = {'AB': 0, 'CD': 1, 'CA': 2, 'CB': 3, 'AD': 4, 'BD': 5}
data['SetSeen '] = data['SetSeen '].map(mapping).astype(int)
data['KeyResponse'] = data['KeyResponse'] - 1

# Divide by condition
baseline = data[data['Condition'] == 'Baseline']
frequency = data[data['Condition'] == 'Frequency']

# convert into dictionary
baseline_dict = dict_generator(baseline)
frequency_dict = dict_generator(frequency)

if __name__ == '__main__':
    # Define the model
    model_delta = ComputationalModels(model_type='delta') # Delta model
    model_delta_PVL = ComputationalModels(model_type='delta_PVL') # Delta-PVL model
    model_delta_asymmetric = ComputationalModels(model_type='delta_asymmetric') # Delta-Asymmetric model
    model_decay = ComputationalModels(model_type='decay') # Decay model
    model_decay_PVL = ComputationalModels(model_type='decay_PVL') # Decay-PVL model
    model_decay_win = ComputationalModels(model_type='decay_win') # Decay-Win model
    model_delta_decay = ComputationalModels(model_type='delta_decay_weight') # Delta-Decay model
    model_delta_decay_PVL = ComputationalModels(model_type='delta_decay_PVL_weight') # DeltaPVL-DecayPVL model
    model_delta_decay_win = ComputationalModels(model_type='delta_decay_win_asymmetric_weight') # Delta-DecayWin model
    model_delta_decay_PVL_win = ComputationalModels(model_type='delta_decay_win_PVL_weight') # DeltaPVL-DecayWin model
    model_delta_asymmetric_decay_win = ComputationalModels(model_type='delta_decay_win_asymmetric_weight') # DeltaAsymmetric-DecayWin model

    model_list = [model_delta, model_delta_PVL, model_delta_asymmetric,
                  model_decay, model_decay_PVL, model_decay_win,
                  model_delta_decay, model_delta_decay_PVL,
                  model_delta_decay_win, model_delta_decay_PVL_win, model_delta_asymmetric_decay_win]

    n_iterations = 100
    model_names = ['delta', 'delta_PVL', 'delta_asymmetric',
                   'decay', 'decay_PVL', 'decay_win',
                   'delta_decay', 'delta_decay_PVL',
                   'delta_decay_win', 'delta_decay_PVL_win', 'delta_asymmetric_decay_win']
    
    # ------------------------------------------------------------------------------------------------------------------
    # Fit Baseline Condition Data
    # (CAREFUL: model fitting can take a long time)
    # ------------------------------------------------------------------------------------------------------------------
    for i, model in enumerate(model_list):
        print(f'Fitting model: {model_names[i]}')
        save_dir = f'./data/ModelFitting/Baseline/{model_names[i]}_results.csv'
        # Check if the file already exists
        try:
         existing_results = pd.read_csv(save_dir)
         if not existing_results.empty:
              print(f"File {save_dir} already exists. Skipping model fitting.")
              continue
        except FileNotFoundError:
         pass

        # fit the model
        model_results = model.fit(baseline_dict, num_training_trials=120, num_iterations=n_iterations,
                                  num_exp_restart=200, initial_mode='first_trial_no_alpha')

        model_results.to_csv(save_dir, index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # Fit Frequency Condition Data
    # (CAREFUL: model fitting can take a long time)
    # ------------------------------------------------------------------------------------------------------------------
    for i, model in enumerate(model_list):
        print(f'Fitting model: {model_names[i]}')
        save_dir = f'./data/ModelFitting/Frequency/{model_names[i]}_results.csv'
        # Check if the file already exists
        try:
         existing_results = pd.read_csv(save_dir)
         if not existing_results.empty:
              print(f"File {save_dir} already exists. Skipping model fitting.")
              continue
        except FileNotFoundError:
         pass

        # fit the model
        model_results = model.fit(frequency_dict, num_training_trials=120, num_iterations=n_iterations,
                                  num_exp_restart=200, initial_mode='first_trial_no_alpha')

        model_results.to_csv(save_dir, index=False)

    # ==================================================================================================================
    # Process Model Fitting Results
    # ==================================================================================================================
    # Define directories and read in the data
    main_model_names = ['delta', 'delta_PVL', 'delta_asymmetric', 'decay', 'decay_PVL', 'decay_win', 'delta_asymmetric_decay_win']
    supp_model_names = ['delta_decay', 'delta_decay_PVL', 'delta_decay_win', 'delta_decay_PVL_win', 'delta_asymmetric_decay_win']
    baseline_results_dir = './data/ModelFitting/Baseline/'
    freq_results_dir = './data/ModelFitting/Frequency/'
    inattentive = pd.read_csv('inattentive_participants.csv')

    # read in the subject data
    def read_results(directory, result_names, inattentive):
        results = {}
        for name in result_names:
            file_path = f'{directory}{name}_results.csv'
            try:
                results[name] = pd.read_csv(file_path)
                results[name]['Model'] = name  # Add a column for the model name
                results[name].rename(columns={'participant_id': 'Subnum'}, inplace=True)
                # remove inattentive participants
                results[name] = results[name][
                    ~results[name]['Subnum'].isin(inattentive['Subnum'].unique())].reset_index(drop=True)
            except FileNotFoundError:
                print(f"File {file_path} not found.")
        return results

    # Read in the subject data and inattentive participants
    E2_baseline_results = read_results(baseline_results_dir, main_model_names, inattentive)
    E2_freq_results = read_results(freq_results_dir, main_model_names, inattentive)

    # ======================================================================================================================
    # Best model
    # ======================================================================================================================
    parameter_map = {
        'delta': ['t', 'alpha'],
        'delta_PVL': ['t', 'alpha', 'scale', 'la'],
        'delta_asymmetric': ['t', 'alpha', 'alpha_neg'],
        'decay': ['t', 'alpha'],
        'decay_PVL': ['t', 'alpha', 'scale', 'la'],
        'decay_win': ['t', 'alpha'],
        'delta_decay': ['t', 'alpha', 'weight'],
        'delta_decay_PVL': ['t', 'alpha', 'scale', 'la', 'weight'],
        'delta_decay_win': ['t', 'alpha', 'weight'],
        'delta_decay_PVL_win': ['t', 'alpha', 'scale', 'la', 'weight'],
        'delta_asymmetric_decay_win': ['t', 'alpha', 'alpha_neg', 'weight']
    }

    model_baseline_bics = []
    for model_name, df in E2_baseline_results.items():
        model_df = df[['Subnum', 'best_parameters', 'AIC', 'BIC']].copy()
        model_df['model'] = model_name
        model_df['Condition'] = 'Baseline'
        model_df = parameter_extractor(model_df, parameter_map[model_name])
        model_baseline_bics.append(model_df)

    model_frequency_bics = []
    for model_name, df in E2_freq_results.items():
        model_df = df[['Subnum', 'best_parameters', 'AIC', 'BIC']].copy()
        model_df['model'] = model_name
        model_df['Condition'] = 'Frequency'
        model_df = parameter_extractor(model_df, parameter_map[model_name])
        model_frequency_bics.append(model_df)

    # Concatenate all models
    all_bics = pd.concat(model_baseline_bics + model_frequency_bics, ignore_index=True)

    # Find the best model per participant
    best_model_per_participant = (
        all_bics.sort_values('BIC')
        .drop_duplicates(['Subnum', 'Condition'], keep='first')
        .reset_index(drop=True)
    )

    # Assuming 'group' is in any of the model DataFrames, like decay
    group_info = E2_baseline_results['delta_asymmetric_decay_win'][['Subnum', 'group_baseline']]
    all_bics = all_bics.merge(group_info, on='Subnum', how='left')
    best_model_per_participant = best_model_per_participant.merge(group_info, on='Subnum', how='left')

    model_counts = (best_model_per_participant.groupby(['Condition', 'model']).size().unstack(fill_value=0)).T

    avg_bic = (all_bics.groupby(['group_baseline', 'Condition', 'model'])['BIC'].mean().unstack(level='model').round(2))

    print(avg_bic)
    print(model_counts)

    # Combine all BIC values into one DataFrame for summary
    model_summary = pd.concat(model_baseline_bics + model_frequency_bics, ignore_index=True)
    model_summary = model_summary.sort_values(['Subnum', 'Condition'], ignore_index=True)
    model_summary.to_csv('./data/model_summary.csv', index=False) # Repeat this process for supplementary models if needed

    # Create a summary statistics table
    summary_list = []

    # iterate through all dataframes in both lists
    for df in itertools.chain(model_frequency_bics, model_baseline_bics):
        model_name = df['model'].iloc[0]

        # keep only numeric columns after dropping id-like cols
        cols_to_drop = ['Subnum', 'Condition', 'model']
        numeric_means = df.drop(columns=cols_to_drop, errors='ignore').select_dtypes('number').mean()

        # attach identifiers
        numeric_means['model'] = model_name
        if 'Condition' in df.columns:
            numeric_means['Condition'] = df['Condition'].iloc[0]

        summary_list.append(numeric_means)

    summary_df = pd.DataFrame(summary_list)
    summary_df['model'] = pd.Categorical(summary_df['model'], categories=main_model_names, ordered=True)
    summary_df = summary_df.sort_values(['Condition', 'model']).reset_index(drop=True)
    summary_df = summary_df.groupby('Condition', group_keys=False).apply(weight_calculation, ['BIC'])
    summary_df = summary_df.round(3)
    summary_df.to_csv('./data/model_summary_statistics.csv', index=False)
    print(summary_df)

    # Calculate Bayes Factor for model comparison
    best_model = 'delta_asymmetric_decay_win'
    bayes_factor_results = []
    condition = model_frequency_bics

    for df in condition:
        model_name = df['model'].iloc[0]
        cond = df['Condition'].iloc[0]
        best_df = get_best_df(condition, best_model, cond)

        BF = bayes_factor(null_results=df, alternative_results=best_df)  # returns a float

        bayes_factor_results.append({
            'Condition': cond,
            'model': model_name,
            f'BF_vs_{best_model}': BF
        })

    bf_table = pd.DataFrame(bayes_factor_results).sort_values(['Condition', 'model']).reset_index(drop=True).round(3)
    print(bf_table)

    # Now calculate variational Bayes indices
    K = 7  # number of models

    # select columns that end with BIC
    dfs = []
    for df in condition:
        model_name = df['model'].iloc[0]
        cond = df['Condition'].iloc[0]
        tmp = df[['Subnum', 'Condition', 'BIC']].copy()
        tmp = tmp.rename(columns={'BIC': f'{model_name}_BIC'})
        dfs.append(tmp)

    combined_BICs = functools.reduce(lambda left, right: pd.merge(left, right, on=['Subnum', 'Condition'], how='outer'),
                                     dfs)
    bic_cols = [col for col in combined_BICs.columns if col.endswith('BIC')]
    log_evidences = combined_BICs[bic_cols].values / (-2)

    # Run VB model selection
    alpha0 = np.ones(K)  # uniform prior
    alpha_est, g_est = vb_model_selection(log_evidences, alpha0=alpha0, tol=1e-12, max_iter=50000)

    # calculate the exceedance probabilities
    ex_probs = compute_exceedance_prob(alpha_est, n_samples=100000)

    # convert all to DataFrame for better readability
    alpha_est_df = pd.DataFrame(alpha_est, index=bic_cols).round(3)
    model_freq = pd.DataFrame((alpha_est / np.sum(alpha_est)).round(3), index=bic_cols, columns=['Frequency'])
    ex_probs_df = pd.DataFrame(ex_probs.round(3), index=bic_cols, columns=['Exceedance Probability'])

    print("Final alpha (Dirichlet parameters):", alpha_est_df.round(3))
    print("Expected model frequencies:", model_freq.round(3))
    print("Exceedance probabilities:", ex_probs_df.round(3))


