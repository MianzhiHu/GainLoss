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

# save the data
data.to_csv('./data/data.csv', index=False)