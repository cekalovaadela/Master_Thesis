'''
This script loads data from TOWT model (df_TOWT.pickle) and generates final comparison plots (XGBoost vs TOWT) based on the
following statistical metrics:
    -> 'mean' = the truth mean value of the building's data
    -> 'mean_xgb'
    -> 'mean_towt'
    -> 'std' = the truth std (standard deviation) value of the building's data
    -> 'std_xgb'
    -> 'std_towt'
    -> 'mse_xgb' (MSE = mean squared error)
    -> 'mse_towt'
    -> 'r2_xgb'
    -> 'r2_towt'
    -> 'cv_rmse_xgb'
    -> 'cv_rmse_towt'
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 24
plt.rcParams["figure.figsize"] = (15,10)

df_compare = pd.read_pickle(".\\data\\df_compare.pickle")

###COMPARISON PLOT of mean values
ax = df_compare[['mean']].plot(style='.', label='truth mean', color=color_pal[9], alpha=0.6, marker='D', markersize=10, figsize=(15, 10))
df_compare['mean_xgb'].plot(ax=ax, style='.', label='XGB mean', color=color_pal[6], alpha=0.6, marker='D', markersize=10)
df_compare['mean_towt'].plot(ax=ax, style='.', label='TOWT mean', color=color_pal[8], alpha=0.6, marker='D', markersize=10)
plt.title('Actual mean vs XGB and TOWT prediction mean')
plt.xlabel('Building index')
plt.ylabel('mean')
plt.legend()
plt.tight_layout()
plt.show()


PROPS_true = {
    'boxprops':{'facecolor':'none', 'edgecolor':color_pal[9]},
    'medianprops':{'color':color_pal[9]},
    'whiskerprops':{'color':color_pal[9]},
    'capprops':{'color':color_pal[9]}
}

PROPS_xgb = {
    'boxprops':{'facecolor':'none', 'edgecolor':color_pal[6]},
    'medianprops':{'color':color_pal[6]},
    'whiskerprops':{'color':color_pal[6]},
    'capprops':{'color':color_pal[6]}
}

PROPS_towt = {
    'boxprops':{'facecolor':'none', 'edgecolor':color_pal[8]},
    'medianprops':{'color':color_pal[8]},
    'whiskerprops':{'color':color_pal[8]},
    'capprops':{'color':color_pal[8]}
}

custom_lines = [Line2D([0], [0], color=color_pal[9], lw=4),
                Line2D([0], [0], color=color_pal[6], lw=4),
                 Line2D([0], [0], color=color_pal[8], lw=4)]

fig, axs = plt.subplots(1, 1, figsize=(10, 10))
fig.suptitle('Actual mean vs XGB and TOWT prediction mean')
sns.boxplot(df_compare['mean'], ax=axs, color=color_pal[9], **PROPS_true)
sns.boxplot(df_compare['mean_xgb'], ax=axs, color=color_pal[6], **PROPS_xgb)
sns.boxplot(df_compare['mean_towt'], ax=axs, color=color_pal[8], **PROPS_towt)
axs.legend(custom_lines, ['true mean', 'xgb mean', 'towt mean'])
plt.tight_layout()
plt.show()


###COMPARISON PLOT of std values
ax = df_compare[['std']].plot(style='.', label='truth STD', color=color_pal[9], alpha=0.6, marker='D', markersize=10, figsize=(15, 10))
df_compare['std_xgb'].plot(ax=ax, style='.', label='XGB STD', color=color_pal[6], alpha=0.6, marker='D', markersize=10)
df_compare['std_towt'].plot(ax=ax, style='.', label='TOWT STD', color=color_pal[8], alpha=0.6, marker='D', markersize=10)
plt.title('Actual STD vs XGB and TOWT prediction STD')
plt.xlabel('Building index')
plt.ylabel('STD')
plt.legend()
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 1, figsize=(10, 10))
fig.suptitle('Actual STD vs XGB and TOWT prediction STD')
sns.boxplot(df_compare['std'], ax=axs, color=color_pal[9], **PROPS_true)
sns.boxplot(df_compare['std_xgb'], ax=axs, color=color_pal[6], **PROPS_xgb)
sns.boxplot(df_compare['std_towt'], ax=axs, color=color_pal[8], **PROPS_towt)
axs.legend(custom_lines, ['true STD', 'xgb STD', 'towt STD'])
plt.tight_layout()
plt.show()


df_compare['mean_xgb_diff'] = abs(df_compare['mean'] - df_compare['mean_xgb'])
df_compare['mean_towt_diff'] = abs(df_compare['mean'] - df_compare['mean_towt'])

ax = plt.bar(df_compare.index, df_compare['mean_xgb_diff'], label='XGB mean diff', color=color_pal[6], alpha=0.6)
plt.bar(df_compare.index, df_compare['mean_towt_diff'], label='TOWT mean diff', color=color_pal[8], alpha=0.6)
plt.title('Differences between truth mean value and predcited mean values by XGB and by TOWT')
plt.xlabel('Building index')
plt.ylabel('mean diff')
plt.legend()
plt.tight_layout()
plt.show()


#Dropping rows with weird values
for i in range(len(df_compare)):
    if (df_compare['r2_towt'][i] <= 0 or df_compare['r2_towt'][i] >= 1 or df_compare['r2_xgb'][i] <= 0 or df_compare['r2_xgb'][i] >= 1):
        print(i)
        df_compare.drop([i], inplace=True)


###COMPARISON PLOT of MSE values
ax = plt.subplots(figsize = (15, 10))
ax = plt.bar(df_compare.index, df_compare['mse_towt'], label='TOWT MSE', color=color_pal[8], alpha=0.6, width=0.8)
plt.bar(df_compare.index, df_compare['mse_xgb'], label='XGB MSE', color=color_pal[6], alpha=0.6, width=0.8)
plt.title('XGB prediction MSE vs TOWT prediction MSE')
plt.xticks(rotation=45)
plt.xlabel('Building index')
plt.ylabel('MSE')
plt.legend()
plt.tight_layout()
plt.show()

###Histograms of MSE
ax1 = plt.subplots(figsize = (15, 10))
ax1 = sns.histplot(df_compare['mse_xgb'], bins=62, kde=True, label='XGB MSE', color=color_pal[6], alpha=0.6)
sns.histplot(df_compare['mse_towt'], bins=62, kde=True, ax=ax1, label='TOWT MSE', color=color_pal[8], alpha=0.6)
plt.xlabel('MSE')
plt.title('Histogram - XGB MSE vs TOWT MSE')
plt.legend()
plt.tight_layout()
plt.show()


###COMPARISON PLOT of r2 values
ax = plt.subplots(figsize = (15, 10))
ax = plt.bar(df_compare.index, df_compare['r2_xgb'], label='XGB R2', color=color_pal[6], alpha=0.6, width=0.8)
plt.bar(df_compare.index, df_compare['r2_towt'], label='TOWT R2', color=color_pal[8], alpha=0.6, width=0.8)
plt.title('XGB prediction R2 vs TOWT prediction R2')
plt.xticks(rotation=45)
plt.xlabel('Building index')
plt.ylabel('R2')
plt.legend()
plt.tight_layout()
plt.show()

###Histograms of r2
ax2 = plt.subplots(figsize = (15, 10))
ax2 = sns.histplot(df_compare['r2_xgb'], bins=62, kde=True, label='XGB R2', color=color_pal[6], alpha=0.6)
sns.histplot(df_compare['r2_towt'], bins=62, kde=True, ax=ax2, label='TOWT R2', color=color_pal[8], alpha=0.6)
plt.xlabel('R2 score')
plt.title('Histogram - XGB R2 vs TOWT R2')
plt.legend()
plt.tight_layout()
plt.show()


###COMPARISON PLOT of CV(RMSE) values
ax3 = plt.subplots(figsize = (15, 10))
ax3 = plt.bar(df_compare.index, df_compare['cv_rmse_towt'], label='TOWT CV(RMSE)', color=color_pal[8], alpha=0.6, width=0.8)
plt.bar(df_compare.index, df_compare['cv_rmse_xgb'], label='XGB CV(RMSE)', color=color_pal[6], alpha=0.6, width=0.8)
plt.title('XGB prediction CV(RMSE) vs TOWT prediction CV(RMSE)')
plt.xticks(rotation=45)
plt.xlabel('Building index')
plt.ylabel('CV(RMSE)')
plt.legend()
plt.tight_layout()
plt.show()

###Histograms of CV(RMSE)
ax4 = plt.subplots(figsize = (15, 10))
ax4 = sns.histplot(df_compare['cv_rmse_xgb'], bins=62, kde=True, label='XGB CV(RMSE)', color=color_pal[6], alpha=0.6)
sns.histplot(df_compare['cv_rmse_towt'], bins=62, ax=ax4, kde=True, label='TOWT CV(RMSE)', color=color_pal[8], alpha=0.6)
plt.xlabel('CV(RMSE)')
plt.title('Histogram - XGB CV(RMSE) vs TOWT CV(RMSE)')
plt.legend()
plt.tight_layout()
plt.show()


df_compare['mse_diff'] = abs(df_compare['mse_xgb'] - df_compare['mse_towt'])
df_compare['r2_diff'] = abs(df_compare['r2_xgb'] - df_compare['r2_towt'])

worst_mse = df_compare.sort_values(by=['mse_diff'], ascending=False).head(15).copy()
worst_r2 = df_compare.sort_values(by=['r2_diff'], ascending=False).head(15).copy()
worst_mse.sort_index(ascending=True, inplace=True)
worst_r2.sort_index(ascending=True, inplace=True)


axis = plt.axes()
plt.bar(range(3, 120, 8), worst_mse['mse_towt'], label='TOWT MSE', color=color_pal[8], alpha=0.6, width=0.8)
plt.bar(range(3, 120, 8), worst_mse['mse_xgb'], label='XGB MSE', color=color_pal[6], alpha=0.6, width=0.8)
plt.title('Buildings with the biggest MSE differences')
axis.set_xticks(range(3, 120, 8))
axis.set_xticklabels(worst_mse.index)
plt.xlabel('Building index')
plt.ylabel('MSE')
plt.legend()
plt.tight_layout()
plt.show()

axis = plt.axes()
ax6 = plt.bar(range(3, 120, 8), worst_r2['r2_xgb'], label='XGB R2', color=color_pal[6], alpha=0.6, width=0.8)
plt.bar(range(3, 120, 8), worst_r2['r2_towt'], label='TOWT R2', color=color_pal[8], alpha=0.6, width=0.8)
plt.title('Buildings with the biggest R2 differences')
axis.set_xticks(range(3, 120, 8))
axis.set_xticklabels(worst_r2.index)
plt.xlabel('Building index')
plt.ylabel('R2')
plt.legend()
plt.tight_layout()
plt.show()


print(f"The mean R2 value of TOWT model: {df_compare['r2_towt'].mean():0.4f}")
print(f"The mean R2 value of XGB model: {df_compare['r2_xgb'].mean():0.4f}")
print(f"The mean MSE value of TOWT model: {df_compare['mse_towt'].mean():0.4f}")
print(f"The mean MSE value of XGB model: {df_compare['mse_xgb'].mean():0.4f}")
print(f"The mean CV(RMSE) value of TOWT model: {df_compare['cv_rmse_towt'].mean():0.4f}")
print(f"The mean CV(RMSE) value of XGB model: {df_compare['cv_rmse_xgb'].mean():0.4f}")


#Finding for which buildings TOWT performed better than XGB
df_compare['r2_winner'] = np.where((df_compare['r2_towt'] > df_compare['r2_xgb']), 'TOWT', 'XGB')
df_compare['mse_winner'] = np.where((df_compare['mse_towt'] < df_compare['mse_xgb']), 'TOWT', 'XGB')

print(df_compare[['mse_xgb', 'mse_towt', 'r2_xgb', 'r2_towt']].loc[df_compare['r2_winner'] == 'TOWT'])
print(df_compare.loc[df_compare['mse_winner'] == 'TOWT'])