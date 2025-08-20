import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Read the data
data = pd.read_csv('./data/E2_data_modeled.csv')
data_summary = pd.read_csv('./data/E2_summary_modeled.csv')
model_summary = pd.read_csv('./data/model_summary.csv')
palette = "deep"
sns.set_theme(style="whitegrid", font="Arial")

# Figure 1a: Training Performance per Condition, Trial Type, and Block
data_1a = data[data['TrialType'].isin(['AB', 'CD'])].groupby(['Subnum', 'Condition', 'TrialType', 'Phase'])['BestOption'].mean().reset_index()

plt.figure(figsize=(10, 4))
palette_1a = {"AB": sns.color_palette(palette)[1], "CD": sns.color_palette(palette)[2]}
g = sns.catplot(data=data_1a, x="Phase", y="BestOption", hue="TrialType", col="Condition", kind="point", dodge=True,
                errorbar=('ci', 95), n_boot=10000, height=4, aspect=1, markers=["o", "s"], linestyles=["--", "-"],
                palette=palette_1a)
g.set_axis_labels("Block", "% of Optimal Choices")
g._legend.set_title("Trial Type", prop={'size': 18})
for text in g._legend.get_texts():
    text.set_fontsize(15)
g._legend.set_bbox_to_anchor((1, 0.3))
g.set_titles("{col_name}", size=20)
for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))  # 2 decimals
    ax.tick_params(axis="both", labelsize=15)
g.set_xlabels(size=18)
g.set_ylabels(size=18)
plt.tight_layout()
plt.savefig('./figures/Figure_1a.png', dpi=600)
plt.show()

# Figure 1b: All Choice Performance per Condition and Trial Type
data_1b = data.groupby(['Subnum', 'Condition', 'TrialType'])['BestOption'].mean().reset_index()
data_1b['TrialType'] = pd.Categorical(data_1b['TrialType'], categories=['AB', 'CD', 'CA', 'CB', 'AD', 'BD'], ordered=True)

plt.figure(figsize=(10, 6))
palette_1b = {"Baseline": sns.color_palette(palette)[0], "Frequency": sns.color_palette(palette)[3]}
sns.barplot(data=data_1b, x="TrialType", y="BestOption", hue="Condition", errorbar=('ci', 95), palette=palette_1b)
plt.axhline(0.5, color=sns.color_palette(palette)[7], linestyle='--', label='Random Chance')
plt.xlabel("Trial Type", fontsize=20)
plt.ylabel("% of Optimal Choices", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(title="Condition", fontsize=18, title_fontsize=20)
plt.tight_layout()
plt.savefig('./figures/Figure_1b.png', dpi=600)
plt.show()

# Figure 1c: Choice Performance in CA trials per Condition
data_1c = data[data['TrialType'] == 'CA'].groupby(['Subnum', 'Condition'])['BestOption'].mean().reset_index()

plt.figure(figsize=(8, 6))
palette_1c = {"Baseline": sns.color_palette(palette)[0], "Frequency": sns.color_palette(palette)[3]}
sns.barplot(data=data_1c, x="Condition", y="BestOption", palette=palette_1c, errorbar=('ci', 95))
plt.axhline(0.5, color=sns.color_palette(palette)[7], linestyle='--', label='Random Chance')
plt.axhline(0.75/(0.65+0.75), color=sns.color_palette(palette)[8], linestyle='-', label='Reward Ratio')
plt.xlabel("Condition", fontsize=20)
plt.ylabel("% of Optimal Choices", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(title="", fontsize=18)
plt.tight_layout()
plt.savefig('./figures/Figure_1c.png', dpi=600)
plt.show()

# Figure 1d: Distribution of C Choices in CA trials per Condition
data_1d = data_1c

plt.figure(figsize=(12, 6))
g = sns.FacetGrid(data_1d, col="Condition", height=5, aspect=1.2)
g.map(sns.histplot, "BestOption", bins=20, kde=True, stat="count", color=sns.color_palette(palette)[5])
g.set_axis_labels("% of C Choices in CA Trials", "N Participants")
g.set_titles("{col_name}", size=20)
for ax in g.axes.flat:
    ax.axvline(0.5, color=sns.color_palette(palette)[7], linestyle='--', label='Random Chance')
    ax.axvline(0.75 / (0.65 + 0.75), color=sns.color_palette(palette)[8], linestyle='-', label='Reward Ratio')
    if ax == g.axes.flat[0]:
        ax.legend(fontsize=15, loc='lower left')
    ax.tick_params(axis="both", labelsize=15)
g.set_xlabels(size=18)
g.set_ylabels(size=18)
plt.tight_layout()
plt.savefig('./figures/Figure_1d.png', dpi=600)
plt.show()

# Figure 2a: best option predicted by training performance per trial type
data_2a = data_summary[data_summary['TrialType'].isin(['CA', 'CB', 'AD', 'BD'])].copy()
data_2a['TrialType'] = pd.Categorical(data_2a['TrialType'], categories=['CA', 'CB', 'AD', 'BD'], ordered=True)

g = sns.lmplot(data=data_2a, x='training_accuracy', y='BestOption', hue='Condition', col='TrialType', markers=[".", "+"],
               palette=palette_1b, height=5, aspect=1, col_wrap=2, scatter_kws={"s": 25, "alpha": 0.4}, legend=False, ci=95)
g.set_axis_labels("Training Accuracy", "% of Optimal Choices", fontsize=20)
g.set_titles("{col_name}", size=20)
ax_ll = g.axes.flatten()[-2]  # second-to-last subplot in FacetGrid = lower-left
leg = ax_ll.legend(loc='lower left', fontsize=18, title='Condition', title_fontsize=20)
leg.get_frame().set_linewidth(0.0)
plt.subplots_adjust(top=0.9)
for ax in plt.gcf().axes:
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
plt.tight_layout()
plt.savefig('./figures/Figure_2a.png', dpi=600)
plt.show()

# Figure 2b: best option in frequency predicted by best option in baseline per trial type
eff = pd.read_csv("./data/eff_df.csv")
if "lower.CL" in eff.columns:
    eff = eff.rename(columns={"lower.CL":"lower","upper.CL":"upper"})
eff["TrialType"] = pd.Categorical(eff["TrialType"], categories=['AB', 'CD', 'CA', 'CB', 'AD', 'BD'], ordered=True)

g = sns.FacetGrid(eff, col="TrialType", col_wrap=2, height=2.5, aspect=1, sharex=True, sharey=True, palette=palette_1b)
for tt, ax in zip(eff["TrialType"].cat.categories, g.axes.flatten()):
    d = eff[eff["TrialType"]==tt]
    ax.fill_between(d["BestOption_Baseline"], d["lower"], d["upper"], alpha=0.15, color=sns.color_palette(palette)[5])
    ax.plot(d["BestOption_Baseline"], d["fit"], linewidth=2, color=sns.color_palette(palette)[5])
for ax in g.axes.flat:
    ax.set_xlabel(""); ax.set_ylabel("")
g.fig.tight_layout(rect=(0.08, 0.06, 0.98, 0.98))  # leave room for big labels
g.fig.supxlabel("% of Optimal Choices in Baseline", fontsize=15, y=0.02)
g.fig.supylabel("Probability of Selecting Optimal Choices in Frequency", fontsize=15, x=0.02)
g.set_titles("{col_name}", size=15)
for ax in g.axes.flatten():
    ax.tick_params(axis="both", labelsize=12)
plt.tight_layout()
plt.savefig("./figures/Figure_2b.png", dpi=600, bbox_inches="tight")
plt.show()

# Figure 3a: Model Fit predicted by Baseline performance
baseline_summary = data_summary[(data_summary['TrialType'] == 'CA') & (data_summary['Condition'] == 'Baseline')].copy()
baseline_summary = baseline_summary.rename(columns={'BestOption': 'Baseline_Performance'})
model_summary = model_summary.merge(baseline_summary[['Subnum', 'Baseline_Performance']], on='Subnum', how='left')
model_summary['model_class'] = np.where(
    model_summary['model'].isin(['delta', 'delta_PVL', 'delta_asymmetric']), 'Delta Rule Models',
    np.where(
        model_summary['model'].isin(['decay', 'decay_PVL', 'decay_win']), 'Decay Rule Models',
        'Hybrid'
    )
)
data_3a = model_summary[(model_summary['Condition'] == 'Frequency') & (model_summary['model_class'].isin(['Delta Rule Models', 'Decay Rule Models']))].copy()
label_map = {
    "delta": "Delta",
    "delta_PVL": "Delta PVL",
    "delta_asymmetric": "Delta Asymmetric",
    "decay": "Decay",
    "decay_PVL": "Decay PVL",
    "decay_win": "Decay Win"
}
data_3a["model_label"] = data_3a["model"].map(label_map)

g = sns.lmplot(data=data_3a, x='Baseline_Performance', y='BIC', col='model_class', hue='model_label',
               markers=["o", "s", "D", "^", "v", "x"], palette=sns.color_palette(palette),
               height=5, aspect=1, scatter=False, legend=True, ci=95)
g.set_axis_labels("Baseline C Choice Rates", "Model Fit (BIC)", fontsize=20)
g.set_titles("{col_name}", size=20)
plt.subplots_adjust(top=0.9)
for ax in plt.gcf().axes:
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
g._legend.set_title("Model", prop={'size': 18})   # legend title size
g._legend.set_bbox_to_anchor((1.05, 0.5))
for text in g._legend.get_texts():
    text.set_fontsize(16)
plt.savefig('./figures/Figure_3a.png', dpi=600, bbox_inches="tight")
plt.show()

# Figure 3b: Weights predicted by Baseline performance
data_3b = model_summary[(model_summary['Condition'] == 'Frequency') & (model_summary['model'] == 'delta_asymmetric_decay_win')].copy()

g = sns.lmplot(data=data_3b, x='Baseline_Performance', y='weight', palette=sns.color_palette(palette), height=5,
               aspect=1, scatter=False,legend=True, ci=95)

g.set_axis_labels("Baseline C Choice Rates", "Value-Based Learning Weight", fontsize=20)
for ax in g.axes.flatten():
    ax.tick_params(axis='both', labelsize=18)
g.fig.subplots_adjust(bottom=0.15, left=0.12)
g.figure.savefig('./figures/Figure_3b.png', dpi=600, bbox_inches='tight')

# Figure 3c: CA choices predicted by parameters
CA_summary = data_summary[data_summary['TrialType'] == 'CA'].copy()
model_summary = model_summary.merge(CA_summary[['Subnum', 'Condition', 'BestOption']], on=['Subnum', 'Condition'], how='left')
data_3c = model_summary.melt(id_vars=['Subnum', 'Condition', 'model', 'model_class', 'BIC', 'BestOption'],
                             value_vars=['t', 'alpha', 'scale', 'la', 'alpha_neg', 'weight'],
                             var_name='Parameter', value_name='Value')

title_map = {
    "t": "Inverse Temperature (c)",
    "alpha": "Learning Rate (α)",
    "scale": "Shape Parameter (γ)",
    "la": "Loss Aversion (λ)",
    "alpha_neg": "Learning Rate - Negative PE (α-)",
    "weight": "Value-Based Learning Weight (w)"
}

g = sns.lmplot(
    data=data_3c,
    x='Value', y='BestOption',
    hue='Condition', col='Parameter', col_wrap=3,
    palette=palette_1c,
    height=5.5, aspect=1.5,
    scatter=False, ci=95,
    legend=True,
    facet_kws={'sharex': False}
)

# Customize x-axis labels and y-axis labels
g.set(xlabel=None)
g.set_ylabels("% of C Choices in CA Trials", fontsize=25)
g.fig.text(0.5, 0.04, "Parameter Value", ha='center', va='center', fontsize=30)

# Apply custom titles
g.set_titles("{col_name}")  # first remove "Parameter = "
for ax, title in zip(g.axes.flatten(), g.col_names):
    ax.set_title(title_map.get(title, title), fontsize=30)

# Adjust tick parameters
for ax in g.axes.flatten():
    ax.tick_params(axis='both', labelsize=25)

# Adjust legend
g._legend.set_title("Condition", prop={'size': 30})
for text in g._legend.get_texts():
    text.set_fontsize(25)

# Adjust spacing and SAVE from the figure (avoid plt.tight_layout/plt.show)
g.fig.subplots_adjust(top=0.9, bottom=0.12, left=0.12, right=0.88, hspace=0.35, wspace=0.20)
g.fig.savefig('./figures/Figure_3c.png', dpi=600, bbox_inches='tight')
plt.close(g.fig)
