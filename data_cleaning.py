import pandas as pd
import numpy as np

# Read in the data
prop_optimal = pd.read_csv('./data/PropOptimal.csv')
data_raw = pd.read_csv('./data/ABCDGainsLossesData_F2023.csv')

# Drop the first column and the last 7 columns
data_raw = data_raw.drop(columns=['Unnamed: 0']).iloc[:, :-8]

# Drop duplicate rows
data_raw = data_raw.drop_duplicates()

# rename the columns
prop_optimal = prop_optimal.rename(columns={'SubjID': 'Subnum'})

# Merge the two dataframes
data = pd.merge(data_raw, prop_optimal, on='Subnum')

# remove the outlier trials (RT > 10s)
# other files don't have this problem
data = data[data['RT'] < 10000]

data.loc[data['PropOptimal'] == 1, 'PropOptimal'] = 0.9999
data.loc[data['PropOptimal'] == 0, 'PropOptimal'] = 0.0001

# in data, if the participants don't have all 6 trials, we need to remove them
data = data.groupby('Subnum').filter(lambda x: len(x) == 6)

# reindex the subnum
data = data.reset_index(drop=True)
data.iloc[:, 0] = (data.index // 6) + 1

# save the data
data.to_csv('./data/data.csv', index=False)