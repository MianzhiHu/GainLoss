import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in the data
data = pd.read_csv('./data/propoptimal_id.csv')

# change the condition to a categorical variable
data.rename(columns={'SetSeen ': 'TrialType'}, inplace=True)

# Rename the trial type
data['TrialType'] = data['TrialType'].replace({0: 'AB', 1: 'CD', 2: 'CA', 3: 'CB', 4: 'AD', 5: 'BD'})

# plot the data
sns.set(style='white')
plt.figure(figsize=(10, 6))
sns.barplot(x='TrialType', y='BestOption', hue='Condition', data=data, errorbar=('ci', 95))
plt.title('Proportion of Optimal Choices')
plt.ylabel('Proportion of Optimal Choices')
plt.xlabel('Trial Type')
# add a horizontal line at 0.5
plt.axhline(0.5, color='darkred', linestyle='--')
sns.despine()
plt.savefig('./figures/propoptimal_all.png', dpi=600)
plt.show()

