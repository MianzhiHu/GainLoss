import pandas as pd
import numpy as np
from utils.Within_Subj_Preprocessing import preprocess_data, extract_numbers

# ======================================================================================================================
# Process Gains and Losses Data
# ======================================================================================================================
# # Read in the data
# prop_optimal = pd.read_csv('./data/PropOptimal.csv')
# data_raw = pd.read_csv('./data/ABCDGainsLossesData_F2023.csv')
# data_2nd_raw = pd.read_csv('./data/ABCDGainsLossesData_F20232ndBatch.csv')
# data_3rd_raw = pd.read_csv('./data/ABCDAllLossesData_F2023_LastBatch.csv')
#
# # combine the data
# gain_losses = pd.concat([data_raw, data_2nd_raw], ignore_index=True)
#
# # reassign the subnum every 250 rows
# gain_losses['Subnum'] = gain_losses.index // 250 + 1
# # gain_losses.to_csv('./data/data.csv', index=False)
#
# # Drop the first column and the last 7 columns
# gain_losses = gain_losses.drop(columns=['Unnamed: 0']).iloc[:, :-7]
#
# # Drop duplicate rows
# gain_losses = gain_losses.drop_duplicates()
#
# # rename the columns
# prop_optimal = prop_optimal.rename(columns={'SubjID': 'Subnum'})
#
# # Merge the two dataframes
# data = pd.merge(gain_losses, prop_optimal, on='Subnum')
#
#
# data.loc[data['PropOptimal'] == 1, 'PropOptimal'] = 0.9999
# data.loc[data['PropOptimal'] == 0, 'PropOptimal'] = 0.0001
#
# # in data, if the participants don't have all 6 trials, we need to remove them
# data = data.groupby('Subnum').filter(lambda x: len(x) == 6)
#
# # # reindex the subnum
# # data = data.reset_index(drop=True)
# # data.iloc[:, 0] = (data.index // 6) + 1
#
# # # save the data
# # data.to_csv('./data/data_PropOptimal.csv', index=False)
#
# data = data[data['Condition'].isin(['Gains', 'GainsEF'])]
# print(data['Subnum'].nunique())

# ======================================================================================================================
# Process Within-Subject Data
# ======================================================================================================================
behavior_var = ['ReactTime', 'Reward', 'BestOption', 'KeyResponse', 'SetSeen ', 'OptionRwdMean']
id_var = ['studyResultId', 'Gender', 'Ethnicity', 'Race', 'Age', 'Big5O', 'Big5C', 'Big5E', 'Big5A', 'Big5N',
          'BISScore', 'CESDScore', 'ESIBF_disinhScore', 'ESIBF_aggreScore', 'ESIBF_sScore', 'PSWQScore',
          'STAITScore', 'STAISScore']
knowledge_var = ['OptionOrder', 'EstOptionA', 'OptionAConfidence', 'EstOptionS', 'OptionSConfidence',
                 'EstOptionK', 'OptionKConfidence', 'EstOptionL', 'OptionLConfidence']
other_var = id_var + knowledge_var
numeric_var = ['ReactTime', 'Reward', 'BestOption', 'KeyResponse', 'SetSeen ', 'OptionRwdMean', 'OptionOrder',
               'EstOptionA', 'OptionAConfidence', 'EstOptionS', 'OptionSConfidence', 'EstOptionK',
               'OptionKConfidence', 'EstOptionL', 'OptionLConfidence', 'studyResultId', 'Big5O', 'Big5C', 'Big5E',
               'Big5A', 'Big5N', 'BISScore', 'CESDScore', 'ESIBF_disinhScore', 'ESIBF_aggreScore', 'ESIBF_sScore',
               'PSWQScore', 'STAITScore', 'STAISScore']
dict_var = ['Gender', 'Ethnicity', 'Race', 'Age']

data_BF, knowledge_BF = preprocess_data('./data/BF', behavior_var, other_var, numeric_var, dict_var,
                                        estimate=True)
data_FB, knowledge_FB = preprocess_data('./data/FB', behavior_var, other_var, numeric_var, dict_var,
                                        baseline_pos=11, compare_pos=0, estimate=True)

# remove knowledge data with from data
data_BF = data_BF.drop(columns=knowledge_var)
data_FB = data_FB.drop(columns=knowledge_var)

# combine the two dataframes
data = pd.concat([data_BF, data_FB], ignore_index=True)
knowledge = pd.concat([knowledge_BF, knowledge_FB], ignore_index=True)

# reset the subnum
subj_id = len(data) // 400 + 1
ids = np.arange(1, subj_id)
ids_behav = np.repeat(ids, 400)
ids_knowledge = np.repeat(ids, 7)
data['Subnum'] = ids_behav
knowledge['Subnum'] = ids_knowledge

# clean the age column
data['Age'] = extract_numbers(data['Age'])

# calculate the proportion of optimal choices
propoptimal = data.groupby(['Subnum', 'Condition', 'SetSeen '])['BestOption'].mean().reset_index()

# save the data
data.to_csv('./data/data_id.csv', index=False)
knowledge.to_csv('./data/knowledge_id.csv', index=False)
propoptimal.to_csv('./data/propoptimal_id.csv', index=False)



