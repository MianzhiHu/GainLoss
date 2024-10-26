import numpy as np
import pandas as pd

# read the model fitting results
result_names = ['delta_id', 'decay_id', 'dual_id', 'delta_baseline_id', 'decay_baseline_id', 'dual_baseline_id',
                'delta_frequency_id', 'decay_frequency_id', 'dual_frequency_id', 'delta_gains', 'decay_gains',
                'dual_gains', 'delta_baseline_gains', 'decay_baseline_gains', 'dual_baseline_gains',
                'delta_frequency_gains', 'decay_frequency_gains', 'dual_frequency_gains']

results = {}
for result_name in result_names:
    results[result_name] = pd.read_csv(f'./data/ModelFitting/{result_name}.csv')
    print(f'[AIC]: {result_name}: {results[result_name]["AIC"].mean()}')
    print(f'[BIC]: {result_name}: {results[result_name]["BIC"].mean()}')
