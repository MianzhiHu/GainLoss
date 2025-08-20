import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import ttest_1samp
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================================================================
# Read the data
# ======================================================================================================================
knowledge = pd.read_csv('./data/knowledge_id.csv')
data = pd.read_csv('./data/E2_data_modeled.csv')
data_summary = pd.read_csv('./data/E2_summary_modeled.csv')

est_cols = ['EstA', 'EstB', 'EstC', 'EstD']

knowledge[est_cols] = knowledge[est_cols].replace([np.inf, -np.inf], np.nan)

# Find the maximum and minimum rewards received (max z-score: 3.7916666666666674; min z-score: -4.270833333333333)
print(f'Maximum Reward Received: {round(data["Reward"].max(), 2)}')
print(f'Minimum Reward Received: {round(data["Reward"].min(), 2)}')

# Remove low-quality rows based on missing values, out-of-range values, and same value across estimates
mask_nan = knowledge[est_cols].isna().any(axis=1)
mask_out_of_range = ((knowledge[est_cols] > 2.09) | (knowledge[est_cols] < -1.09)).any(axis=1)
mask_same_value = (knowledge[est_cols].nunique(axis=1, dropna=True) == 1)

# # Conservative approach: we remove the participants who have at least one low-quality row
# participants_to_drop = knowledge[mask_nan | mask_same_value | mask_out_of_range]['Subnum'].unique()
# knowledge = knowledge[~knowledge['Subnum'].isin(participants_to_drop)]
# print(knowledge['Subnum'].nunique())
# print(knowledge.shape)

# Liberal approach: we only remove low-quality rows while keeping the participants
rows_to_drop = knowledge[mask_nan | mask_same_value | mask_out_of_range].index.unique()
knowledge = knowledge[~knowledge.index.isin(rows_to_drop)]
print(knowledge['Subnum'].nunique())
print(knowledge.shape)

# Apply confidence to the estimates
knowledge['EstA_weighted'] = knowledge['EstA'] * knowledge['A_Confidence']
knowledge['EstB_weighted'] = knowledge['EstB'] * knowledge['B_Confidence']
knowledge['EstC_weighted'] = knowledge['EstC'] * knowledge['C_Confidence']
knowledge['EstD_weighted'] = knowledge['EstD'] * knowledge['D_Confidence']

# Normalize the estimates
knowledge[['EstA_normalized', 'EstB_normalized', 'EstC_normalized', 'EstD_normalized']] = (
    knowledge[['EstA_weighted','EstB_weighted','EstC_weighted','EstD_weighted']].apply(
        lambda row: pd.Series((row.values / row.values.sum()) * 2), axis=1))

# Calculate CA difference based on 6th phase
knowledge['CA_diff'] = knowledge['EstC'] - knowledge['EstA']
knowledge['CA_diff_normalized'] = knowledge['EstC_normalized'] - knowledge['EstA_normalized']
knowledge_CA = knowledge[knowledge['Phase'] == 6][['Subnum', 'Condition', 'CA_diff', 'CA_diff_normalized']].copy().reset_index(drop=True)

# Convert the knowledge DataFrame to long format for estimates and confidence
knowledge_estimate = pd.melt(knowledge, id_vars=['Subnum', 'Condition', 'Phase'],
                         value_vars=est_cols, var_name='Option', value_name='Value')

knowledge_conf = pd.melt(knowledge, id_vars=['Subnum', 'Condition', 'Phase'],
                         value_vars=['A_Confidence', 'B_Confidence', 'C_Confidence', 'D_Confidence'],
                         var_name='Option', value_name='Confidence')

knowledge_normalized = pd.melt(knowledge, id_vars=['Subnum', 'Condition', 'Phase'],
                               value_vars=['EstA_normalized', 'EstB_normalized', 'EstC_normalized', 'EstD_normalized'],
                               var_name='Option', value_name='Value')

# Strip prefix/suffix so only A/B/C/D remain
knowledge_estimate['Option'] = knowledge_estimate['Option'].str.extract(r'Est([ABCD])')[0]
knowledge_conf['Option'] = knowledge_conf['Option'].str.extract(r'([ABCD])_Confidence')[0]
knowledge_normalized['Option'] = knowledge_normalized['Option'].str.extract(r'Est([ABCD])_normalized')[0]

# Calculate the distance between the estimates and the EV for each option
EV_map = {
    'A': 0.65,
    'B': 0.35,
    'C': 0.75,
    'D': 0.25
}

knowledge_estimate['err'] = knowledge_estimate['Value'] - knowledge_estimate['Option'].map(EV_map)
knowledge_estimate['abs_err'] = knowledge_estimate['err'].abs()

# Calculate phase-wise accuracy
# If phase > 7, phase = 7
data['Phase'] = data['Phase'].clip(upper=7)
data_phase = data.groupby(['Subnum', 'Condition', 'Phase'])['BestOption'].mean().reset_index()

CA_summary = data_summary[data_summary['TrialType'] == 'CA'][['Subnum', 'Condition', 'BestOption', 'training_accuracy']].copy()


# Combine dataframes
knowledge_data = pd.merge(knowledge_estimate, data_phase, on=['Subnum', 'Condition', 'Phase'])
knowledge_data = pd.merge(knowledge_data, knowledge_conf, on=['Subnum', 'Condition', 'Phase', 'Option'])
knowledge_data = pd.merge(knowledge_data, knowledge_CA, on=['Subnum', 'Condition'], how='left')
knowledge_data = pd.merge(knowledge_data, knowledge_normalized, on=['Subnum', 'Condition', 'Phase', 'Option'], suffixes=('', '_normalized'))
knowledge_data = pd.merge(knowledge_data, CA_summary, on=['Subnum', 'Condition'], how='left', suffixes=('', '_CA'))
knowledge_data.to_csv('./data/knowledge_summary.csv', index=False)
knowledge_data[knowledge_data['Phase'] == 6].to_csv('./data/knowledge_summary_phase7.csv', index=False)

# Repeated-measures ANOVA
aov = pg.rm_anova(
    dv='Value',
    within=['Condition', 'Option'],
    subject='Subnum',
    data=knowledge_data,
    detailed=True
)
print(aov)

# Post-hoc tests
posthoc = pg.pairwise_ttests(
    dv='Value',
    within=['Condition', 'Option'],
    subject='Subnum',
    data=knowledge_data,
    padjust='fdr_bh',
    effsize='cohen'
)
print(posthoc)

# Correlation between CA difference between conditions
# Keep only participants who have CA difference in both conditions
knowledge_CA = knowledge_CA.dropna(subset=['CA_diff'])
knowledge_CA = knowledge_CA.groupby('Subnum').filter(lambda x: len(x) == 2)
ca_diff_corr = pg.corr(
    x=knowledge_CA[knowledge_CA['Condition'] == 'Baseline']['CA_diff'],
    y=knowledge_CA[knowledge_CA['Condition'] == 'Frequency']['CA_diff'],
    method='pearson'
)

# mediation analysis
mediation_data = knowledge_data[knowledge_data['Phase'] == 6].drop_duplicates()
mediation_data = mediation_data[mediation_data['Option'].isin(['C', 'A'])].copy()
mediation_data['CA_diff_abs'] = mediation_data['CA_diff'].abs()
mediation_data.to_csv('./data/mediation_data.csv', index=False)

mediation_data_copy = mediation_data[['Subnum', 'Condition', 'training_accuracy', 'CA_diff', 'BestOption_CA']].copy().drop_duplicates()
baseline_copy = mediation_data_copy[mediation_data_copy['Condition'] == 'Baseline']
frequency_copy = mediation_data_copy[mediation_data_copy['Condition'] == 'Frequency']
med = pg.mediation_analysis(frequency_copy, x='training_accuracy', m='CA_diff', y='BestOption_CA')
print(med)

med_model = smf.ols(
    'CA_diff_abs ~ training_accuracy * Condition',
    data=mediation_data
).fit()
print(med_model.summary())

out_model = smf.ols(
    'BestOption_CA ~ training_accuracy * Condition + CA_diff_abs * Condition',
    data=mediation_data
).fit()
print(out_model.summary())

# Extract coefficients
a_base = med_model.params['training_accuracy']                     # effect on CA_diff in Baseline
a_freq = a_base + med_model.params.get('training_accuracy:Condition[T.Frequency]', 0)

b_base = out_model.params['CA_diff_abs']                          # effect of CA_diff on C_choice_CA in Baseline
b_freq = b_base + out_model.params.get('CA_diff_abs:Condition[T.Frequency]', 0)

indirect_base = a_base * b_base
indirect_freq = a_freq * b_freq

print(f"Indirect effect (Baseline): {indirect_base:.4f}")
print(f"Indirect effect (Frequency): {indirect_freq:.4f}")


# Set the style for seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
g = sns.FacetGrid(knowledge_estimate, col="Condition", hue="Option", height=5, aspect=1.2)
g.map(sns.lineplot, "Phase", "Value", marker="o")
g.add_legend()
plt.tight_layout()
plt.savefig('./figures/knowledge.png', dpi=600)
plt.show()

# # Set the style for seaborn
# sns.set(style="whitegrid")
# # Create a figure and axis
# plt.figure(figsize=(10, 6))
# g = sns.FacetGrid(knowledge_conf, col="Condition", hue="Option", height=5, aspect=1.2)
# g.map(sns.lineplot, "Phase", "Confidence", marker="o")
# g.add_legend()
# plt.tight_layout()
# plt.savefig('./figures/knowledge.png', dpi=600)
# plt.show()
