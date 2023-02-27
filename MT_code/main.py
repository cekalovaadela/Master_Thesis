import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
from datetime import timedelta

import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, TimeSeriesSplit, train_test_split)

import xgboost as xgb
from xgboost import plot_importance, plot_tree

from load_data import Merger
from modify_data import DF, Figure, Modifier, Model
from read_data import Reader

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

plt.rcParams['font.size'] = 24


# Merger().merge_files() #merges all csv files (used only once at the beginning to create data.pickle)
data = Reader().read_data() #loads data.pickle files with all the input data for all 124 builidngs

i = 22  #setting the index of the building that is to be analyzed
df, df_name = DF().create_df(data, i)
df.index = pd.to_datetime(df.index)

###CREATING NaN DataFrame for plotting purposes
df_nan = df.copy()
df_nan = pd.DataFrame(columns=df.columns, index=df.index)

###DROPPING all rows that have missing values
df = Modifier().drop_missing_val(df)

###EXTRACTING datetime features from timestamp 'ts'
df = DF().add_day_time(df)

###LabelEncoder - Converting categorical variables (occ) to dummy indicators ('False' -> 0, 'True' -> 1)
#Buildings with only "True" one value for occupancy: 1, 5, 6, 7, 14, 15
df = Modifier().encode_labels(df)

###HOLIDAYS - Setting occupancy to 0 for public holidays for buildings that are not constantly in operation (they have more than 20% of unoccupied days in the dataset)
holidays_df = DF().create_holiday_df() #creates a DataFrame with all czech public holidays from the year 2018 to 2022
df = Modifier().set_holiday_unoccupancy(df, holidays_df)

###DEALING WITH OUTLIERS
# df_sorted = df.sort_values(by = ('meas'), ascending=False)  
# print(df_sorted)

###HISTOGRAMS and BOX PLOT before outlier removal
fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.set_title('Histogram - input power')
sns.histplot(df['meas'], bins=20, kde=True, ax=axs, label='meas')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='month', y='meas')
ax.set_title('meas by month')
plt.tight_layout()
plt.show()


###REMOVING OUTLIERS based on interquartile range
df = Modifier().drop_outliers(df)

###HISTOGRAMS and BOX PLOT after outlier removal
fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.set_title('Histogram - input power')
sns.histplot(df['meas'], bins=20, kde=True, ax=axs, label='meas')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='month', y='meas')
ax.set_title('meas by month')
plt.tight_layout()
plt.show()
# plt.savefig('boxplot_22.png', transparent=True)

###SCALING - using MinMaxScaler - For each feature, each value is subtracted by the minimum value of the respective feature 
# and then divide by the range of original maximum and minimum of the same feature. It has a default range between [0,1].
df.drop(['datetime', 'date'], axis=1, inplace=True)
df = Modifier().scale(df)

###ADDING AVERAGE FEATURES - these features can be only used for analysis (e.g. anomaly detection), not for future predictions
df = DF().add_features(df)

###LAG FEATURES - what was the target variable 'meas' (x) days in the past and that information is fed to the model as a new feature
df = DF().add_lags(df) #creates features 'lag_1year' and 'lag_1week'


###Creating mask DataFrame for plotting purposes
df_mask = df_nan.combine_first(df[['occ', 'temp', 'meas']])


###USING HEATMAP to find correlations (Pearson correlation coefficients) between features and drop the correlated ones
Figure().plot_heatmap(df)  #shows intial heatmaps before dropping correlated features
df.drop(['quarter', 'weekofyear', 'dayofyear'], axis=1, inplace=True) #dropping these columns based on the heatmap - very correlated with 'month'
Figure().plot_heatmap(df)  #heatmap after dropping correlated features


###PAIRPLOTS of the whole df is pretty big and confusing since a lot of data is there => plotting only certain features
cols_to_plot = ['occ', 'temp', 'meas', 'hour', 'dayofweek', 'month']
sns.pairplot(df[cols_to_plot], hue='occ')
plt.tight_layout()
plt.show()



# ### TIME SERIES CROSS VALIDATION - could be used to evaluate the model
#------------------------------------------------------------------------
# test_len = int(len(df)*0.1)
# tss = TimeSeriesSplit(n_splits=4, test_size=test_len, gap=24)  # test_size is now set to 10% of the length of the dataset
# # gap is set to 24 hours (for dataset when there around hourly data) - 24 hours between the training set ends and the test set begins
# df = df.sort_index()

# fig, axs = plt.subplots(4, 1, figsize=(15, 15), sharex=True)

# fold = 0
# for train_idx, val_idx in tss.split(df):
#     train = df.iloc[train_idx]
#     test = df.iloc[val_idx]
#     train['meas'].plot(ax=axs[fold],
#                           label='Training Set')
#     test['meas'].plot(ax=axs[fold],
#                          label='Test Set')
#     axs[fold].axvline(test.index.min(), color='black', ls='--', linewidth=3)
#     axs[fold].set_title(f'Data Train/Test Split Fold {fold}', fontsize=12)
#     fold += 1
# plt.subplots_adjust(hspace=0.25)
# plt.show()
#--------------------------------------------------------------------------


###DIVIDING DATASET INTO DEPENDENT (TARGET) AND INDEPENDENT VARIABLES
#X ... features, y ... target variable (energy consumption)
X = df.drop(['meas'], axis=1)
y = df['meas']


###USING train_test_split with some percentage to split dataset into train and test datasets - 30 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=False)

split_date = y_test.index[0]
split_date = "'" + str(split_date) + "'"


###Only for plotting purposes
X_train_mask = df_mask[['temp', 'occ']].loc[df_mask[['temp', 'occ']].index < split_date].copy()
X_test_mask = df_mask[['temp', 'occ']].loc[df_mask[['temp', 'occ']].index >= split_date].copy()
y_train_mask = df_mask['meas'].loc[df_mask['meas'].index < split_date].copy()
y_test_mask = df_mask['meas'].loc[df_mask['meas'].index >= split_date].copy()


###Uncomment to plot separate graphs for temperature and meas; otherwise, they are plottetd in the next step together with occ
# fig, ax = plt.subplots(1, 1, figsize=(15, 7), sharex=True)
# X_train_mask[['temp']].plot(style='-', ax=ax, color=color_pal[0], label='train_temp', title='Data Train/Test Split')
# X_test_mask[['temp']].plot(style='-', ax=ax, color=color_pal[1], label='test_temp')
# ax.axvline(split_date, color='black', ls='--')
# ax.legend(['train_temp', 'test_temp'])
# ax.set_ylabel('Temperature [°C]')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(15, 7), sharex=True)
# y_train_mask.plot(style='-', ax=ax, color=color_pal[3], label='train_meas', title='Data Train/Test Split')
# y_test_mask.plot(style='-', ax=ax, color=color_pal[9], label='train_meas')
# ax.axvline(split_date, color='black', ls='--')
# ax.legend(['train_meas', 'test_meas'])
# ax.set_ylabel('Input power [kW]')
# plt.tight_layout()
# plt.show()


###PLOTTING THE WHOLE DATASET WITH TRAIN/TEST SPLIT
fig, ax = plt.subplots(3, 1, figsize=(15, 25), sharex=True)
X_train_mask[['temp']].plot(style='-', ax=ax[0], color=color_pal[0], label='train_temp', title='Data Train/Test Split')
X_test_mask[['temp']].plot(style='-', ax=ax[0], color=color_pal[1], label='test_temp')
ax[0].axvline(split_date, color='black', ls='--')
ax[0].legend(['train_temp', 'test_temp'])
ax[0].set_ylabel('T [°C]')
y_train_mask.plot(style='-', ax=ax[1], color=color_pal[3], label='train_meas')
y_test_mask.plot(style='-', ax=ax[1], color=color_pal[9], label='train_meas')
ax[1].axvline(split_date, color='black', ls='--')
ax[1].legend(['train_meas', 'test_meas'])
ax[1].set_ylabel('Meas [kW]')
X_train_mask[['occ']].plot(style='.', ax=ax[2], color=color_pal[2], label='train_occ')
X_test_mask[['occ']].plot(style='.', ax=ax[2], color=color_pal[4], label='test_occ')
ax[2].axvline(split_date, color='black', ls='--')
ax[2].legend(['train_occ', 'test_occ'])
ax[2].set_ylabel('Occ')
plt.tight_layout()
plt.show()


###GradientBoostingRegressor cannot handle lag features because they have some NaN values
X_train_clf = X_train.drop(['lag_1year', 'lag_1week'], axis=1)
X_test_clf = X_test.drop(['lag_1year', 'lag_1week'], axis=1)


###SKLEARN - performs slightly worse than XGBRegressor
clf = GradientBoostingRegressor(loss='squared_error',
                                learning_rate=0.01, 
                                n_estimators=1000, 
                                subsample=0.65,
                                max_depth=3,  
                                random_state=None)

clf.fit(X_train_clf, y_train)
print(f'\nGradientBoostingRegressor has accuracy of: {clf.score(X_test_clf, y_test):0.4f}')

clf_mse = mean_squared_error(y_test, clf.predict(X_test_clf))
print(f'\nThe mean squared error (MSE) on test set: {clf_mse:0.4f}')


###XGB
reg = xgb.XGBRegressor(base_score=0.5,
                       tree_method = 'gpu_hist',
                       predictor='gpu_predictor', 
                       booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:squarederror',
                       max_depth=3,
                       subsample=0.65,
                       learning_rate=0.01)

reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)

print(f'\nXGBRegressor has accuracy of: {reg.score(X_test, y_test):0.4f}')

mse = mean_squared_error(y_test, reg.predict(X_test))
print(f'\nThe mean squared error (MSE) on test set: {mse:0.4f}')

###FEATURE IMPORTANCE PLOT
#Available importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
importance_gain = reg.get_booster().get_score(importance_type='gain')
print(importance_gain)
print(reg.get_booster().get_score(importance_type='weight'))
for key in importance_gain.keys():
    importance_gain[key] = round(importance_gain[key],2)
plt.rcParams["figure.figsize"] = (10,10)
xgb.plot_importance(importance_gain, height=0.6)
plt.tight_layout()
plt.show()

###DECISION TREES PLOTTING
fig, ax = plt.subplots(figsize=(30, 30))
xgb.plot_tree(reg, num_trees=0, ax=ax) #num_trees is the index of the tree in the ensemble that should be plotted
plt.tight_layout()
plt.show()


###GRID SEARCH for XGBRegressor (too time-consuming)
#-----------------------------------------------------------
# xgb_grid = xgb.XGBRegressor(objective= 'reg:squarederror', tree_method = 'gpu_hist', predictor='gpu_predictor', eval_metric=mean_squared_error, early_stopping_rounds=50)
# parameters = {
#     'n_estimators': [700, 1000, 1400],
#     'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
#     'max_depth': [3, 5, 7, 10],
#     'reg_alpha': [1.1, 1.2, 1.3],
#     'reg_lambda': [1.1, 1.2, 1.3],
#     'subsample': [0.6, 0.7, 0.8],
#     'learning_rate': [0.01, 0.05, 0.1]}

# fit_params={"eval_set" : [[X_test, y_test]]}

# cv = 4

# grid_search = GridSearchCV(
#     estimator=xgb_grid,
#     param_grid=parameters,
#     scoring = 'neg_mean_squared_error',
#     n_jobs = -1,
#     cv = TimeSeriesSplit(n_splits=cv).get_n_splits([X_train, y_train]),
#     verbose=False)

# xgb_grid_model = grid_search.fit(X_train, y_train, **fit_params, verbose=1000)

# print('Best Parameter:')
# print(xgb_grid_model.best_params_) 
# print('\n------------------------------------------------------------------\n')
# print(xgb_grid_model.best_estimator_)

# reg = xgb_grid_model.best_estimator_

# reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
# print(f'\nXGBRegressor has accuracy of: {reg.score(X_test, y_test):0.4f}')

# mse = mean_squared_error(y_test, reg.predict(X_test))
# print(f'\nThe mean squared error (MSE) on test set: {mse:0.4f}')
#-----------------------------------------------------------


###PREDICTING TARGET VARIABLES - XGB
y_pred = reg.predict(X_test)
df_pred = pd.DataFrame({'pred_meas': y_pred})
df_pred.index = X_test.index

df = df.merge(df_pred, how='left', left_index=True, right_index=True)
df_mask = df_mask.merge(df_pred, how='left', left_index=True, right_index=True)



####PLOTTING ACTUAL MEAS vs PREDICTED MEAS
ax = df_mask[['meas']].plot(style='-', label='truth meas', color=color_pal[9], alpha=0.7, figsize=(15, 5))
df_mask['pred_meas'].plot(ax=ax, style='-', label='predicted meas', color=color_pal[6], alpha=0.6)
first_pred = df['pred_meas'].first_valid_index()
first_day = first_pred.date() + datetime.timedelta(1)
last_pred = df['pred_meas'].last_valid_index()
print(f"First predicted day {first_pred} with meas value of {df['pred_meas'].loc[first_pred]:0.4f}")
print(f"Last predicted day {last_pred} with meas value of {df['pred_meas'].loc[last_pred]:0.4f}")
first_pred = "'" + str(first_pred) + "'"
last_pred = "'" + str(last_pred) + "'"
ax.set_xbound(lower=first_pred, upper=last_pred)
ax.set_ylabel('Input power')
plt.title('Actual meas vs predicted meas')
plt.legend()
plt.show()

###Finding the first Monday in the predicted values and protting the first week of predictions
first_monday = DF().next_weekday(first_day, 0) # 0 = Monday, 1=Tuesday, 2=Wednesday...
next_monday = first_monday + datetime.timedelta(7)

first_monday = "'" + str(first_monday) + "'"
next_monday = "'" + str(next_monday) + "'"

###PLOTTING the first week of predictions starting from Monday
ax = df_mask[['meas']].plot(style='-', label='truth meas', color=color_pal[9], alpha=0.8, figsize=(15, 5))
df_mask['pred_meas'].plot(ax=ax, style='-', label='predicted meas', color=color_pal[6], alpha=0.7)
ax.set_xbound(lower=first_monday, upper=next_monday)
ax.set_ylabel('Input power')
plt.title('First week of predictions (starting from Monday)')
plt.legend()
plt.show()


###EVALUATION METRICS ON TEST SET
df_eval = df[['occ', 'temp', 'meas', 'pred_meas']].loc[df.index >= split_date].copy()

Model().evaluation_metrics(df_eval=df_eval[['meas', 'pred_meas']], y_true=df_eval['meas'], y_pred=df_eval['pred_meas'])

###FINDING THE WORST AND THE BEST PREDICTED DAYS
df_eval['error'] = df_eval['meas'] - df_eval['pred_meas']
df_eval['abs_error'] = df_eval['error'].abs()
df_eval = DF().add_day_time(df_eval)
error_by_day = df_eval.groupby(['dayofmonth', 'month', 'year']).mean()[['meas', 'pred_meas', 'abs_error']]
print(f'The worst predicted days: \n{error_by_day.sort_values(by=["abs_error"], ascending=False).head(10)}')
print(f'The best predicted days: \n{error_by_day.sort_values(by=["abs_error"], ascending=True).head(10)}')