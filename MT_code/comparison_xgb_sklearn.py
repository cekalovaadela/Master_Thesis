'''
This script loads data from scikit-learn GradientBoostingRegressor model (sklearn_compare.pickle) and from xgboost (XGBRegressor) model and
compares both models' performance.
Each data frame includes R2 scores and MSE values.
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sklearn_compare = pd.read_pickle(".\\data\\sklearn_compare.pickle")
xgb_compare = pd.read_pickle(".\\data\\xgb_compare.pickle")

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 24
plt.rcParams["figure.figsize"] = (15,10)


#Dropping rows with weird values
for i in range(len(sklearn_compare)):
    if (sklearn_compare['R2_sklearn'][i] <= 0 or sklearn_compare['R2_sklearn'][i] >= 1 or xgb_compare['R2_xgb'][i] <= 0 or xgb_compare['R2_xgb'][i] >= 1):
        print(i)
        sklearn_compare.drop([i], inplace=True)
        xgb_compare.drop([i], inplace=True)


###XGB vs sklearn R2 scores
ax = plt.subplots(figsize = (15, 5))
ax = plt.bar(sklearn_compare.index, sklearn_compare['R2_sklearn'], label='SKLEARN R2', color=color_pal[1], alpha=0.6, width=0.8)
plt.bar(xgb_compare.index, xgb_compare['R2_xgb'], label='XGB R2', color=color_pal[6], alpha=0.6, width=0.8)
plt.title('XGB R2 score vs SKLEARN R2 score')
plt.xticks(rotation=45)
plt.xlabel('Building index')
plt.ylabel('R2')
plt.legend()
plt.tight_layout()
plt.show()

###Histograms of R2
ax= sns.histplot(sklearn_compare['R2_sklearn'], bins=62, kde=True, label='SKLEARN R2', color=color_pal[1], alpha=0.6)
sns.histplot(xgb_compare['R2_xgb'], bins=62, kde=True, ax=ax, label='XGB R2', color=color_pal[6], alpha=0.6)
plt.xlabel('R2 score')
plt.title('Histogram - XGB R2 vs SKLEARN R2')
plt.legend()
plt.tight_layout()
plt.show()


###XGB vs sklearn MSE
ax = plt.subplots(figsize = (15, 5))
ax = plt.bar(sklearn_compare.index, sklearn_compare['mse_sklearn'], label='SKLEARN MSE', color=color_pal[1], alpha=0.6, width=0.8)
plt.bar(xgb_compare.index, xgb_compare['mse_xgb'], label='XGB MSE', color=color_pal[6], alpha=0.6, width=0.8)
plt.title('XGB MSE vs SKLEARN MSE')
plt.xticks(rotation=45)
plt.xlabel('Building index')
plt.ylabel('MSE')
plt.legend()
plt.tight_layout()
plt.show()

###Histograms of MSE
ax= sns.histplot(sklearn_compare['mse_sklearn'], bins=62, kde=True, label='SKLEARN MSE', color=color_pal[1], alpha=0.6)
sns.histplot(xgb_compare['mse_xgb'], bins=62, kde=True, ax=ax, label='XGB MSE', color=color_pal[6], alpha=0.6)
plt.xlabel('MSE')
plt.title('Histogram - XGB MSE vs SKLEARN MSE')
plt.legend()
plt.tight_layout()
plt.show()


print(f"The mean R2 value of sklearn model: {sklearn_compare['R2_sklearn'].mean():0.4f}")
print(f"The mean R2 value of XGB model: {xgb_compare['R2_xgb'].mean():0.4f}")
print(f"The mean MSE value of sklearn model: {sklearn_compare['mse_sklearn'].mean():0.4f}")
print(f"The mean MSE value of XGB model: {xgb_compare['mse_xgb'].mean():0.4f}")


df_compare = sklearn_compare.copy()
df_compare = df_compare.merge(xgb_compare, how='left', left_index=True, right_index=True)

df_compare['R2_diff'] = abs(sklearn_compare['R2_sklearn'] - xgb_compare['R2_xgb'])
df_compare['mse_diff'] = abs(sklearn_compare['mse_sklearn'] - xgb_compare['mse_xgb'])

worst_mse = df_compare.sort_values(by=['mse_diff'], ascending=False).head(15).copy()
worst_r2 = df_compare.sort_values(by=['R2_diff'], ascending=False).head(15).copy()
worst_mse.sort_index(ascending=True, inplace=True)
worst_r2.sort_index(ascending=True, inplace=True)


axis = plt.axes()
plt.bar(range(3, 120, 8), worst_mse['mse_sklearn'], label='SKLEARN MSE', color=color_pal[1], alpha=0.6, width=0.8)
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
plt.bar(range(3, 120, 8), worst_r2['R2_xgb'], label='XGB R2', color=color_pal[6], alpha=0.6, width=0.8)
plt.bar(range(3, 120, 8), worst_r2['R2_sklearn'], label='SKLEARN R2', color=color_pal[1], alpha=0.6, width=0.8)
plt.title('Buildings with the biggest R2 differences')
axis.set_xticks(range(3, 120, 8))
axis.set_xticklabels(worst_r2.index)
plt.xlabel('Building index')
plt.ylabel('R2')
plt.legend()
plt.tight_layout()
plt.show()


#Finding for which buildings SKLEARN performed better than XGB
df_compare['r2_winner'] = np.where((df_compare['R2_sklearn'] > df_compare['R2_xgb']), 'SKLEARN', 'XGB')
df_compare['mse_winner'] = np.where((df_compare['mse_sklearn'] < df_compare['mse_xgb']), 'SKLEARN', 'XGB')

print(df_compare.loc[df_compare['r2_winner'] == 'SKLEARN'])
print(df_compare.loc[df_compare['mse_winner'] == 'SKLEARN'])