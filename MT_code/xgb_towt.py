'''
This script loads data from TOWT model (df_TOWT.pickle) and generates comparison prediction plots (XGBoost vs TOWT) for each building.
New data frame including statistical measure is created so that it can be used for statistical comparison of models.
Following statistical measures are included:
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
import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

from modify_data import DF, Modifier, Model
from read_data import Reader


df_TOWT = pd.read_pickle(".\\data\\df_TOWT.pickle")

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 24

data = Reader().read_data() #loads data.pickle files with all the input data for all 124 builidngs

df_compare = pd.DataFrame()

for i in range(len(data)):
    df, df_name = DF().create_df(data, i)
    df.index = pd.to_datetime(df.index)

    df_nan = df.copy()
    df_nan = pd.DataFrame(columns=df.columns, index=df.index)

    df = Modifier().drop_missing_val(df)

    df = DF().add_day_time(df)

    df = Modifier().encode_labels(df)

    holidays_df = DF().create_holiday_df() 
    df = Modifier().set_holiday_unoccupancy(df, holidays_df)

    df = Modifier().drop_outliers(df)

    df.drop(['datetime', 'date'], axis=1, inplace=True)

    df = Modifier().scale(df)

    df = DF().add_features(df)

    df = DF().add_lags(df)

    df_mask = df_nan.combine_first(df[['occ', 'temp', 'meas']])

    df.drop(['quarter', 'weekofyear', 'dayofyear'], axis=1, inplace=True) #dropping these columns based on the heatmap - very correlated with 'month'

    X = df.drop(['meas'], axis=1)
    y = df['meas']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=False)

    split_date = y_test.index[0]
    split_date = "'" + str(split_date) + "'"

    X_train_mask = df_mask[['temp', 'occ']].loc[df_mask[['temp', 'occ']].index < split_date].copy()
    X_test_mask = df_mask[['temp', 'occ']].loc[df_mask[['temp', 'occ']].index >= split_date].copy()
    y_train_mask = df_mask['meas'].loc[df_mask['meas'].index < split_date].copy()
    y_test_mask = df_mask['meas'].loc[df_mask['meas'].index >= split_date].copy()


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


    ###PREDICTING TARGET VARIABLES - XGB
    y_pred = reg.predict(X_test)
    df_pred = pd.DataFrame({'pred_meas_xgb': y_pred})
    df_pred.index = X_test.index

    ###ADDING predictions from TOWT model for comparison
    TOWT_cols = [col for col in df_TOWT.columns if df_name in col]

    dfTOWT = df_TOWT[TOWT_cols]
    dfTOWT = dfTOWT.set_index(TOWT_cols[0])

    df_pred = df_pred.merge(dfTOWT, how='left', left_index=True, right_index=True)
    df_pred.rename(columns = {TOWT_cols[1]:'pred_meas_towt'}, inplace = True)

    df = df.merge(df_pred, how='left', left_index=True, right_index=True)

    df_mask = df_mask.merge(df_pred, how='left', left_index=True, right_index=True)


    ###PLOTTING ACTUAL MEAS vs PREDICTED MEAS (whole dataset, only test set, specific week of a month -> whatever I want)
    ax = df_mask[['meas']].plot(style='-', label='truth meas', color=color_pal[9], alpha=0.7, figsize=(15, 6))
    df_mask['pred_meas_xgb'].plot(ax=ax, style='-', label='XGB meas', color=color_pal[6], alpha=0.6)
    df_mask['pred_meas_towt'].plot(ax=ax, style='-', label='TOWT meas', color=color_pal[8], alpha=0.4)
    first_pred = df['pred_meas_xgb'].first_valid_index()
    first_day = first_pred.date() + datetime.timedelta(1)
    last_pred = df['pred_meas_xgb'].last_valid_index()
    first_pred = "'" + str(first_pred) + "'"
    last_pred = "'" + str(last_pred) + "'"
    ax.set_xbound(lower=first_pred, upper=last_pred)
    plt.title('Actual meas vs predicted meas')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'data_exploration\\TOWT_vs_XGB_prediction\\towt_xgb_{i}.png')
    plt.close()

    ###Finding the first Monday in the predicted values and protting the first week of predictions
    first_monday = DF().next_weekday(first_day, 0) # 0 = Monday, 1=Tuesday, 2=Wednesday...
    next_monday = first_monday + datetime.timedelta(7)

    first_monday = "'" + str(first_monday) + "'"
    next_monday = "'" + str(next_monday) + "'"


    ###PLOTTING first week of predictions starting from Monday
    ax = df_mask[['meas']].plot(style='-', label='truth meas', color=color_pal[9], alpha=0.8, figsize=(15, 6))
    df_mask['pred_meas_xgb'].plot(ax=ax, style='-', label='XGB meas', color=color_pal[6], alpha=0.7)
    df_mask['pred_meas_towt'].plot(ax=ax, style='-', label='TOWT meas', color=color_pal[8], alpha=0.4)
    ax.set_xbound(lower=first_monday, upper=next_monday)
    plt.title('Actual vs predicted meas - first week')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'data_exploration\\TOWT_vs_XGB_prediction_week\\towt_xgb_week_{i}.png')
    plt.close()


    ###EVALUATION METRICS ON TEST SET
    df_eval = df[['occ', 'temp', 'meas', 'pred_meas_xgb', 'pred_meas_towt']].loc[df.index >= split_date].copy()


    df_compare1 = pd.DataFrame({'mean': [np.mean(df_eval['meas'], axis=0)],
                                'mean_xgb': np.mean(df_eval['pred_meas_xgb']),
                                'mean_towt': np.mean(df_eval['pred_meas_towt']),
                                'std': np.std(df_eval['meas']),
                                'std_xgb': np.std(df_eval['pred_meas_xgb']),
                                'std_towt': np.std(df_eval['pred_meas_towt']),
                                'mse_xgb': mean_squared_error(df_eval['meas'], df_eval['pred_meas_xgb']),
                                'mse_towt': mean_squared_error(df_eval['meas'], df_eval['pred_meas_towt']),
                                'r2_xgb': r2_score(df_eval['meas'], df_eval['pred_meas_xgb']),
                                'r2_towt': r2_score(df_eval['meas'], df_eval['pred_meas_towt']),
                                'cv_rmse_xgb': (np.sqrt(mean_squared_error(df_eval['meas'], df_eval['pred_meas_xgb'])) / np.mean(df_eval["meas"], axis=0)),
                                'cv_rmse_towt': (np.sqrt(mean_squared_error(df_eval['meas'], df_eval['pred_meas_towt'])) / np.mean(df_eval["meas"], axis=0))
                                })


    df_compare = df_compare.append(df_compare1, ignore_index=True)
    print(df_compare)

    with open("data_exploration/TOWT_vs_XGB.txt", "a") as file:   
            file.write(f'\n\nBuilding {df_name} with index {i}:')

    Model().write_evaluation_metrics(df_eval=df_eval[['meas', 'pred_meas_xgb']], y_true=df_eval['meas'], y_pred=df_eval['pred_meas_xgb'])
    Model().write_evaluation_metrics(df_eval=df_eval[['meas', 'pred_meas_towt']], y_true=df_eval['meas'], y_pred=df_eval['pred_meas_towt'])

df_compare.to_pickle("C:\\Users\\adm\\Desktop\\Master_Thesis\\MT_code\\data\\df_compare.pickle")

