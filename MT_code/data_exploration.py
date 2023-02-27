import matplotlib.pyplot as plt
import pandas as pd
from pandas_profiling import ProfileReport 
import seaborn as sns
from scipy.stats import spearmanr
import datetime

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import xgboost as xgb
from xgboost import plot_tree

from load_data import Merger
from modify_data import DF, Figure, Modifier, Model
from read_data import Reader

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 24

# Merger().merge_files() #merges all csv files (used only once at the beginning to create data.pickle)
data = Reader().read_data() #loads data.pickle files with all the input data for all 124 builidngs

start = 0 #By setting different start values, distinct parts of the script will be executed

if start == 1:
    '''
    Generates ProfileReports for each df (building) - it is a html file with info about the df
        -> Number of variables - Variable types
        -> Number of observations
        -> Missing cells
        -> Missing cells (%)
        -> Duplicate rows
        -> Duplicate rows (%)
        -> Total size in memory
        -> Average record size in memory
        -> then info about each variable (feature) - distinct values, missing, mean, min, max, zeros, negative, distribution, interactions between variables, correlation between variables
    Generates histograms and boxplots before and after outliers removal and saves them to data_exploration folder
    Generates heatmaps and pairplots to recognize the correlated features
    Plots the whole dataset for each building including the train/test split
    Plots graphs of predictions - whole the whole test set and for the first week of predictions
    Plots feature importance
    Plots the first decision tree for each ensemble
    Logs evaluation metrics and compares GradientBoostingRegressor with XGBRegressor in log_data_exploration.txt
    '''
    for i in range(len(data)):

        df, df_name = DF().create_df(data, i)
        df.index = pd.to_datetime(df.index)

        ###CREATING NaN DataFrame for plotting purposes
        df_nan = df.copy()
        df_nan = pd.DataFrame(columns=df.columns, index=df.index)

        ###GENERATING REPORT 
        df_report = DF().add_day_time(df).copy()

        df_report.drop(['datetime'], axis=1, inplace=True)
        profile_explorative = ProfileReport(df_report, title = 'DataFrame Profile Report', html = {'style': {'full_width': True }}, explorative=True) 
        profile_explorative.to_file(f'data_exploration\\df_report\\df_report_{i}.html')

        ###DROPPING all rows that have missing values
        df = Modifier().drop_missing_val(df)

        ###EXTRACTING datetime features from timestamp 'ts'
        df = DF().add_day_time(df)

        ###LabelEncoder - Converting categorical variables (occ) to dummy indicators
        df = Modifier().encode_labels(df)

        ###HOLIDAYS - Setting occupancy to 0 for public holidays for buildings that are not constantly in operation (they have more than 20% of unoccupied days in the dataset)
        holidays_df = DF().create_holiday_df() #creates a DataFrame with all czech public holidays from the year 2018 to 2022
        df = Modifier().set_holiday_unoccupancy(df, holidays_df)

        ###HISTOGRAMS before outlier removal
        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f'Histograms')
        sns.histplot(df['temp'], bins=20, kde=True, ax=axs[0], label='temp')
        sns.histplot(df['meas'], bins=20, kde=True, ax=axs[1], label='meas')
        plt.xticks(rotation = 45)
        plt.tight_layout()
        plt.savefig(f'data_exploration\\histograms_outliers\\hist_plot_{i}.png')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=df, x='month', y='meas')
        ax.set_title('meas by month')
        plt.savefig(f'data_exploration\\boxplots_outliers\\box_plot_{i}.png')
        plt.close()

        ###REMOVING OUTLIERS based on interquartile range
        df = Modifier().drop_outliers(df)

        ###HISTOGRAMS after outlier removal
        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f'Histograms')
        sns.histplot(df['temp'], bins=20, kde=True, ax=axs[0], label='temp')
        sns.histplot(df['meas'], bins=20, kde=True, ax=axs[1], label='meas')
        plt.xticks(rotation = 45)
        plt.tight_layout()
        plt.savefig(f'data_exploration\\histograms_no_outliers\\histogram_{i}.png')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=df, x='month', y='meas')
        ax.set_title('meas by month')
        plt.tight_layout()
        plt.savefig(f'data_exploration\\boxplots_no_outliers\\boxplot_{i}.png')
        plt.close()

        ###SCALING - using MinMaxScaler
        df.drop(['datetime', 'date'], axis=1, inplace=True)
        df = Modifier().scale(df)

        ###ADDING FEATURES - these features can be only used for analysis (e.g. anomaly detection), not for future predictions
        df = DF().add_features(df)

        ###LAG FEATURES - what was the target variable 'meas' (x) days in the past and that information is fed to the model as a new feature
        # df = DF().add_lags(df) #bc GradientBoostingRegressor cannot handle lag features because they have some NaN values

        #Creating mask DataFrame for plotting purposes
        df_mask = df_nan.combine_first(df[['occ', 'temp', 'meas']])
        
        ###Using HEATMAP to find correlations (Pearson correlation coefficients) between features
        colormap = plt.cm.RdBu
        plt.figure(figsize=(18, 18))
        sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=False, cmap=colormap, 
                        linecolor='white', annot=True) # getting either close to 1 or -1 means some sort of strong linear correlation
        plt.title('Pearson Correlation of Features')
        plt.xticks(rotation = 10)
        plt.savefig(f'data_exploration\\heatmaps_before\\heatmap_b_{i}.png')
        plt.close()

        df.drop(['quarter', 'weekofyear', 'dayofyear'], axis=1, inplace=True) #dropping these columns based on the heatmap - very correlated with 'month'

        plt.figure(figsize=(18, 18))
        sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=False, cmap=colormap, 
                        linecolor='white', annot=True) # getting either close to 1 or -1 means some sort of strong linear correlation
        plt.title('Pearson Correlation of Features')
        plt.xticks(rotation = 10)
        plt.savefig(f'data_exploration\\heatmaps_after\\heatmap_a_{i}.png')
        plt.close()

        ###PAIRPLOTS of the whole df is pretty big and confusing since a lot of data is there => maybe considering throwing away some columns (features)
        cols_to_plot = ['occ', 'temp', 'meas', 'hour', 'dayofweek', 'month']
        sns.pairplot(df[cols_to_plot], diag_kind='kde', diag_kws={'color':'red'})
        plt.tight_layout()
        plt.savefig(f'data_exploration\\pairplots\\pairplot_{i}.png')
        plt.close()

        sns.pairplot(df[cols_to_plot], hue='occ')
        plt.tight_layout()
        plt.savefig(f'data_exploration\\pairplots\\pairplot_occ_{i}.png')
        plt.close()

        ###DIVIDING DATASET INTO DEPENDENT (TARGET) AND INDEPENDENT VARIABLES
        #X ... features, y ... target variable (energy consumption)
        X = df.drop(['meas'], axis=1)
        y = df['meas']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=False)

        split_date = y_test.index[0]
        split_date = "'" + str(split_date) + "'"

        #Only for plotting purposes
        X_train_mask = df_mask[['temp', 'occ']].loc[df_mask[['temp', 'occ']].index < split_date].copy()
        X_test_mask = df_mask[['temp', 'occ']].loc[df_mask[['temp', 'occ']].index >= split_date].copy()
        y_train_mask = df_mask['meas'].loc[df_mask['meas'].index < split_date].copy()
        y_test_mask = df_mask['meas'].loc[df_mask['meas'].index >= split_date].copy()


        ### PLOTTING THE WHOLE DATASET WITH TRAIN/TEST SPLIT
        fig, ax = plt.subplots(3, 1, figsize=(15, 25), sharex=True)
        X_train_mask[['temp']].plot(style='-', ax=ax[0], color=color_pal[0], label='train_temp', title='Data Train/Test Split')
        X_test_mask[['temp']].plot(style='-', ax=ax[0], color=color_pal[1], label='test_temp')
        ax[0].axvline(split_date, color='black', ls='--')
        ax[0].legend(['train_temp', 'test_temp'])
        y_train_mask.plot(style='-', ax=ax[1], color=color_pal[3], label='train_meas')
        y_test_mask.plot(style='-', ax=ax[1], color=color_pal[9], label='train_meas')
        ax[1].axvline(split_date, color='black', ls='--')
        ax[1].legend(['train_meas', 'test_meas'])
        X_train_mask[['occ']].plot(style='.', ax=ax[2], color=color_pal[2], label='train_occ')
        X_test_mask[['occ']].plot(style='.', ax=ax[2], color=color_pal[4], label='test_occ')
        ax[2].axvline(split_date, color='black', ls='--')
        ax[2].legend(['train_occ', 'test_occ'])
        plt.savefig(f'data_exploration\\dataset_plot\\dataset_plot_{i}.png')
        plt.close()

        ###SKLEARN - performs slightly worse than XGBRegressor
        clf = GradientBoostingRegressor(loss='squared_error',
                                        learning_rate=0.01, 
                                        n_estimators=1000, 
                                        subsample=0.65,
                                        max_depth=3,  
                                        random_state=None)
        clf.fit(X_train, y_train)
        clf_mse = mean_squared_error(y_test, clf.predict(X_test))
        
        ###XGB
        reg = xgb.XGBRegressor(base_score=0.5,
                            tree_method = 'gpu_hist',
                            predictor='gpu_predictor', 
                            booster='gbtree',    
                            n_estimators=1000,
                            early_stopping_rounds=50,
                            objective='reg:squarederror',
                            #    objective='reg:pseudohubererror',
                            max_depth=3,
                            subsample=0.65,
                            learning_rate=0.01)
        reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
        mse = mean_squared_error(y_test, reg.predict(X_test))


        ##FEATURE IMPORTANCE PLOT
        xgb.plot_importance(reg, height=0.6)
        plt.tight_layout()
        plt.savefig(f'data_exploration\\feature_importance\\feature_plot_{i}.png')
        plt.close()

        ###DECISION TREES PLOTTING
        plot_tree(reg, num_trees=0) #num_trees is the index of the tree in the ensemble that should be plotted
        plt.tight_layout()
        plt.savefig(f'data_exploration\\decision_trees\\decision_tree0_{i}.png')
        plt.close()

        ###PREDICTING TARGET VARIABLES - XGB
        y_pred = reg.predict(X_test)
        df_pred = pd.DataFrame({'pred_meas': y_pred})
        df_pred.index = X_test.index
    
        df = df.merge(df_pred, how='left', left_index=True, right_index=True)

        df_mask = df_mask.merge(df_pred, how='left', left_index=True, right_index=True)

        ####PLOTTING ACTUAL MEAS vs PREDICTED MEAS (whole dataset, only test set, specific week of a month -> whatever I want)
        ax = df_mask[['meas']].plot(style='-', label='truth meas', color=color_pal[9], alpha=0.7, figsize=(15, 5))
        df_mask['pred_meas'].plot(ax=ax, style='-', label='predicted meas', color=color_pal[6], alpha=0.6)
        first_pred = df['pred_meas'].first_valid_index()
        first_day = first_pred.date() + datetime.timedelta(1)
        last_pred = df['pred_meas'].last_valid_index()
        first_pred = "'" + str(first_pred) + "'"
        last_pred = "'" + str(last_pred) + "'"
        ax.set_xbound(lower=first_pred, upper=last_pred)
        plt.title('Actual meas vs predicted meas')
        plt.legend()
        plt.savefig(f'data_exploration\\prediction_plot\\prediction_plot_{i}.png')
        plt.close()

        #Finding the first Monday in the predicted values and protting the first week of predictions
        first_monday = DF().next_weekday(first_day, 0) # 0 = Monday, 1=Tuesday, 2=Wednesday...
        next_monday = first_monday + datetime.timedelta(7)

        first_monday = "'" + str(first_monday) + "'"
        next_monday = "'" + str(next_monday) + "'"

        #Plotting first week of predictions starting from Monday
        ax = df_mask[['meas']].plot(style='-', label='truth meas', color=color_pal[9], alpha=0.8, figsize=(15, 5))
        df_mask['pred_meas'].plot(ax=ax, style='-', label='predicted meas', color=color_pal[6], alpha=0.7)
        ax.set_xbound(lower=first_monday, upper=next_monday)
        plt.title('Actual meas vs predicted meas - first week (starting from Monday)')
        plt.legend()
        plt.savefig(f'data_exploration\\prediction_week\\prediction_week_{i}.png')
        plt.close()

        ###EVALUATION METRICS ON TEST SET
        df_eval = df[['occ', 'temp', 'meas', 'pred_meas']].loc[df.index >= split_date].copy()
        Model().evaluation_metrics(df_eval=df_eval[['meas', 'pred_meas']], y_true=df_eval['meas'], y_pred=df_eval['pred_meas'])

        ######### FINDING THE WORST AND THE BEST PREDICTED DAYS
        df_eval['error'] = df_eval['meas'] - df_eval['pred_meas']
        df_eval['abs_error'] = df_eval['error'].abs()
        df_eval = DF().add_day_time(df_eval)
        error_by_day = df_eval.groupby(['dayofmonth', 'month', 'year']).mean()[['meas', 'pred_meas', 'abs_error']]

        ###OTHER CORRELATION METRICS
        spearmanr_coefficient, p_value = spearmanr(df['occ'], df['dayofweek'])


        with open("data_exploration/log_data_exploration.txt", "a") as file:
            # Writing data to a file - "append mode"
            file.write(f'\nBuilding {df_name} with index {i}:')
            file.write(f'\n\nFirst day of df: {df.index[0]} \nLast day of df: {df.index[-1]} \nLength of df: {len(df)}')
            file.write(f'\n\nGradientBoostingRegressor has accuracy of: {clf.score(X_test, y_test):0.4f}')
            file.write(f'\nMSE of GradientBoostingRegressor: {mse:0.4f}')
            file.write(f'\n\nXGBRegressor has accuracy of: {reg.score(X_test, y_test):0.4f}')
            file.write(f'\nMSE of XGBRegressor: {mse:0.4f}')
            file.write(f"\n\nFirst prediction: {first_pred} \n\tActual meas value: {df['meas'].loc[first_pred]:0.4f} \n\tPredicted meas value: {df['pred_meas'].loc[first_pred]:0.4f}")
            file.write(f"\n\nLast prediction: {last_pred} \n\tActual meas value: {df['meas'].loc[last_pred]:0.4f} \n\tPredicted meas value: {df['pred_meas'].loc[last_pred]:0.4f}")
            file.write(f"\n\nNumber of prediction made: {len(y_pred)}")
            file.write(f"\n\nFirst week of predictions (starting from Monday): {first_monday} -> {next_monday}")
            file.write(f'\n\nThe worst predicted days: \n{error_by_day.sort_values(by=["abs_error"], ascending=False).head(10)}')
            file.write(f'\n\nThe best predicted days: \n{error_by_day.sort_values(by=["abs_error"], ascending=True).head(10)}')
            file.write(f"\n\nSpearman Rank Correlation Coefficient ('occ' vs 'dayofweek') {spearmanr_coefficient:0.3f}")
            file.write(f'\n-----------------------------------------------------------------------------------------------\n\n\n')




elif start == 2:
    '''
    Generates pie plots and statistics for column "occ" (occ_pie folder)
    Analyzes occupied and unoccupied days and whether the unoccupied weekdays fall on a public holiday and save the results to log_data_occupancy.txt
    '''
    for i in range(1):
        df, df_name = DF().create_df(data, 7)

        df.index = pd.to_datetime(df.index)

        df = Modifier().drop_missing_val(df)

        df = DF().add_day_time(df)

        ## Creating pie plots for occupancy, but some bulidings don't have both 'True' and 'False' values so it has to be taken care of that
        if len(df['occ'].value_counts().index) == 2:
            plt.pie(df['occ'].value_counts(), autopct='%.1f')
            label1 = str(df['occ'].value_counts().index[0])
            label2 = str(df['occ'].value_counts().index[1])
            plt.legend([label1, label2], loc='best')
            plt.title(f"Occupancy counts: \n{label1}: {df['occ'].value_counts().values[0]} \n{label2}: {df['occ'].value_counts().values[1]}", size=15, loc='center')
            plt.tight_layout()
            plt.savefig(f'data_exploration\occ_pie\occ_pie_{i}.png', transparent=True)
            plt.close()

        elif len(df['occ'].value_counts().index) == 1:

            if str(df['occ'].value_counts().index[0]) == 'True':
                plt.pie(df['occ'].value_counts(), autopct='%.1f')
                label = str(df['occ'].value_counts().index[0])
                plt.legend([label], loc='best')
                plt.title(f"Occupancy counts: \n{label}: {df['occ'].value_counts().values[0]}", size=15, loc='center')
                plt.tight_layout()
                plt.savefig(f'data_exploration\occ_pie\occ_pie_{i}.png', transparent=True)
                plt.close()

            elif str(df['occ'].value_counts().index[0]) == 'False': 
                plt.pie(df['occ'].value_counts(), autopct='%.1f')
                label = str(df['occ'].value_counts().index[0])
                plt.legend([label], loc='best')
                plt.title(f"Occupancy counts: \n{label}: {df['occ'].value_counts().values[0]}", size=15, loc='center')
                plt.tight_layout()
                plt.savefig(f'data_exploration\occ_pie\occ_pie_{i}.png', transparent=True)
                plt.close()

            else:
                print('Pie inside else')
        else:
            print('Pie outside else')

        ## LabelEncoder - Converting categorical variables (occ) to dummy indicators
        df = Modifier().encode_labels(df)

        with open("data_exploration/log_data_occupancy.txt", "a", encoding="utf-8") as file:
            file.write(f'\nBuilding {df_name} with index {i}:')
            n = len(pd.unique(df['minute']))
            if n > 1:
                file.write(f'\n\n15-min interval data ({n} unique values in "minute" column)')
            else:
                file.write(f'\n\n1-hour interval data ({n} unique values in "minute" column)')

            unoccupied_weekend = []
            unoccupied_weekdays = []

            for j in range(len(df)):
                if df['occ'][j] == 0:  #unoccupied
                    if df['dayofweek'][j] == 6 or df['dayofweek'][j] == 7:
                        unoccupied_weekend.append(df['date'][j])
                    else:
                        unoccupied_weekdays.append(df['date'][j])

            weekend_len = len(pd.unique(unoccupied_weekend))
            weekdays_len = len(pd.unique(unoccupied_weekdays))

            df_weekdays = pd.DataFrame(unoccupied_weekdays, columns=['unocc_weekdays'])
            df_weekend = pd.DataFrame(unoccupied_weekend)
            unique_weekdays = pd.unique(df_weekdays['unocc_weekdays'])
            unique_weekdays = pd.DataFrame(unique_weekdays, columns=['unique_weekdays']).set_index('unique_weekdays')
            
            holidays_df = DF().create_holiday_df()       

            file.write(f'\n\nFirst day of df: {df.index[0]} \nLast day of df: {df.index[-1]} \nLength of df: {len(df)}')
            file.write(f'\n\nNumber of unique days in data set: {len(pd.unique(df["date"]))}')
            file.write(f"\n\nValues counts for column 'occ': \n{df['occ'].value_counts()}")
            file.write(f'\n\nCount of unoccupied weekend days: {weekend_len}')
            file.write(f'\n\nCount of unoccupied week days: {weekdays_len} ')
            file.write(f'\n\nTotal count of unoccupied days {weekend_len} + {weekdays_len} = {weekend_len+weekdays_len}')
            file.write(f'\n\nUnoccupied weekdays: {unique_weekdays.index}')
            for i in range(len(unique_weekdays)):
                un_weekday = "'" + str(unique_weekdays.index[i]) + "'"
                file.write(f'\n\n{un_weekday}')
                if un_weekday in holidays_df.index:
                    holiday_name_cz = holidays_df['hol_name_cz'].loc[un_weekday]
                    holiday_name_en = holidays_df['hol_name_en'].loc[un_weekday]
                    holiday = holidays_df[['hol_name_cz', 'hol_name_en']].loc[un_weekday]
                    file.write(f'\t{holiday_name_cz}')
                    file.write(f'\t{holiday_name_en}')
                else:
                    file.write(f'\tNot holiday!')
            file.write(f'\n-----------------------------------------------------------------------------------------------\n\n\n')





elif start == 3:
    '''
    Identifying builidngs with weird temperatures 
    -> finds the indices of buildings that have temperature values our=tside of this range (-25°, 50°)
    '''
    idx = []
    for i in range(len(data)):
        df, df_name = DF().create_df(data, i)

        df.index = pd.to_datetime(df.index)
        df['datetime'] = df.index

        df = Modifier().drop_missing_val(df)


        building_idx = []
        building_datetime = []
        building_temp = []

        for j in range(len(df)):
            if (df['temp'][j] > 50 or df['temp'][j] < -25):
                building_idx.append(i)
                building_datetime.append(df['datetime'][j])
                building_temp.append(df['temp'][j])

        building_data = pd.DataFrame({'Building_idx': building_idx, 'datetime': building_datetime, 'temp_value': building_temp})
        
        idx.append(building_idx)

    unique_idx = {x for l in idx for x in l}
    print(unique_idx)





elif start == 4:
    '''
    Comparing model's performance when only feature df['meas_dayofweek_avg'] (folder - meas_dayofweek_avg.txt) 
    or when both df['meas_month_avg'] and df['meas_dayofweek_avg'] (folder - meas_month_avg and meas_dayofweek_avg.txt) are added
    '''
    for i in range(len(data)):
        df, df_name = DF().create_df(data, i)

        df.index = pd.to_datetime(df.index)

        df = Modifier().drop_missing_val(df)

        df = DF().add_day_time(df)

        df = Modifier().encode_labels(df)

        holidays_df = DF().create_holiday_df()
        df = Modifier().set_holiday_unoccupancy(df, holidays_df)

        ###REMOVING OUTLIERS based on interquartile range
        df = Modifier().drop_outliers(df)

        ###SCALING - using MinMaxScaler
        df.drop(['datetime', 'date'], axis=1, inplace=True)
        df = Modifier().scale(df)

        #Adding features
        df['meas_month_avg'] = df.groupby(['month', 'year'])['meas'].transform('mean')
        df['meas_dayofweek_avg'] = df.groupby(['dayofweek', 'month', 'year'])['meas'].transform('mean') #calculating average meas for a dayofweek in a specific month

        #Dropping correlated features
        df.drop(['quarter', 'weekofyear', 'dayofyear'], axis=1, inplace=True) #dropping these columns based on the heatmap - very correlated with 'month'

        X = df.drop(['meas'], axis=1)
        y = df['meas']  

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=False)

        split_date = y_test.index[0]
        split_date = "'" + str(split_date) + "'"


        reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                            n_estimators=1000,
                            early_stopping_rounds=50,
                            objective='reg:squarederror',
                            max_depth=3,
                            learning_rate=0.01)
        reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)


        with open("data_exploration/meas_month_avg and meas_dayofweek_avg.txt", "a", encoding="utf-8") as file:
            file.write(f'\n\nBuilding {df_name} with index {i}:')
            file.write(f'\nXGBRegressor has accuracy of: {reg.score(X_test, y_test):0.4f}')

            mse = mean_squared_error(y_test, reg.predict(X_test))
            file.write(f'\nThe mean squared error (MSE) on test set: {mse:0.4f}')


    for i in range(len(data)):
        df, df_name = DF().create_df(data, i)

        df.index = pd.to_datetime(df.index)

        df = Modifier().drop_missing_val(df)

        df = DF().add_day_time(df)

        df = Modifier().encode_labels(df)

        holidays_df = DF().create_holiday_df() #creates a DataFrame with all czech public holidays from the year 2018 to 2022
        df = Modifier().set_holiday_unoccupancy(df, holidays_df)

        ###REMOVING OUTLIERS based on interquartile range
        df = Modifier().drop_outliers(df)

        ###SCALING - using MinMaxScaler
        df.drop(['datetime', 'date'], axis=1, inplace=True)
        df = Modifier().scale(df)

        #Adding features
        df['meas_dayofweek_avg'] = df.groupby(['dayofweek', 'month', 'year'])['meas'].transform('mean') #calculating average meas for a dayofweek in a specific month

        #Dropping correlated features
        df.drop(['quarter', 'weekofyear', 'dayofyear'], axis=1, inplace=True) #dropping these columns based on the heatmap - very correlated with 'month'

        X = df.drop(['meas'], axis=1)
        y = df['meas']  

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=False)

        split_date = y_test.index[0]
        split_date = "'" + str(split_date) + "'"

        reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                            n_estimators=1000,
                            early_stopping_rounds=50,
                            objective='reg:squarederror',
                            max_depth=3,
                            learning_rate=0.01)
        reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)

    
        with open("data_exploration/meas_dayofweek_avg.txt", "a", encoding="utf-8") as file:
            file.write(f'\n\nBuilding {df_name} with index {i}:')
            file.write(f'\nXGBRegressor has accuracy of: {reg.score(X_test, y_test):0.4f}')

            mse = mean_squared_error(y_test, reg.predict(X_test))
            file.write(f'\nThe mean squared error (MSE) on test set: {mse:0.4f}')





elif start == 5:
    '''
    Comparing accuracy of GradientBoostingRegressor (sklearn) and XGBRegressor
    '''

    sklearn_compare = pd.DataFrame()
    xgb_compare = pd.DataFrame()

    for i in range(len(data)):
        df, df_name = DF().create_df(data, i)

        df.index = pd.to_datetime(df.index)

        df = Modifier().drop_missing_val(df)

        df = DF().add_day_time(df)

        df = Modifier().encode_labels(df)

        holidays_df = DF().create_holiday_df()
        df = Modifier().set_holiday_unoccupancy(df, holidays_df)

        ###REMOVING OUTLIERS based on interquartile range
        df = Modifier().drop_outliers(df)

        ###SCALING - using MinMaxScaler
        df.drop(['datetime', 'date'], axis=1, inplace=True)
        df = Modifier().scale(df)

        #Adding features
        df['meas_dayofweek_avg'] = df.groupby(['dayofweek', 'month', 'year'])['meas'].transform('mean') #calculating average meas for a dayofweek in a specific month

        #Dropping correlated features
        df.drop(['quarter', 'weekofyear', 'dayofyear'], axis=1, inplace=True) #dropping these columns based on the heatmap - very correlated with 'month'

        X = df.drop(['meas'], axis=1)
        y = df['meas']  

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=False)

        split_date = y_test.index[0]
        split_date = "'" + str(split_date) + "'"

        ###SKLEARN 
        clf = GradientBoostingRegressor(loss='squared_error',
                                        learning_rate=0.01, 
                                        n_estimators=1000, 
                                        subsample=0.65,
                                        max_depth=3,  
                                        random_state=None)

        clf.fit(X_train, y_train)

        mse_clf = mean_squared_error(y_test, clf.predict(X_test))

        sklearn_compare1 = pd.DataFrame({'R2_sklearn': [clf.score(X_test, y_test)],
                                    'mse_sklearn': [mean_squared_error(y_test, clf.predict(X_test))]})

        sklearn_compare = sklearn_compare.append(sklearn_compare1, ignore_index=True)
        print(sklearn_compare)
        
        
        ###XGB
        reg = xgb.XGBRegressor(base_score=0.5,    
                            n_estimators=1000,
                            early_stopping_rounds=50,
                            objective='reg:squarederror',
                            max_depth=3,
                            subsample=0.65,
                            learning_rate=0.01)
        
        reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)

        mse_reg = mean_squared_error(y_test, reg.predict(X_test))

        xgb_compare1 = pd.DataFrame({'R2_xgb': [reg.score(X_test, y_test)],
                                    'mse_xgb': [mean_squared_error(y_test, reg.predict(X_test))]})

        xgb_compare = xgb_compare.append(xgb_compare1, ignore_index=True)
        print(xgb_compare)
        

        with open("data_exploration/SKLEARN_vs_XGB.txt", "a", encoding="utf-8") as file:
            file.write(f'\n\n\nBuilding {df_name} with index {i}:')

            file.write(f'\nGradientBoostingRegressor has accuracy of: {clf.score(X_test, y_test):0.4f}')
            file.write(f'\nMSE of GradientBoostingRegressor: {mse_clf:0.4f}')

            file.write(f'\n\nXGBRegressor has accuracy of: {reg.score(X_test, y_test):0.4f}')
            file.write(f'\nMSE of XGBRegressor: {mse_reg:0.4f}')

    sklearn_compare.to_pickle("C:\\Users\\adm\\Desktop\\Master_Thesis\\MT_code\\data\\sklearn_compare.pickle")
    xgb_compare.to_pickle("C:\\Users\\adm\\Desktop\\Master_Thesis\\MT_code\\data\\xgb_compare.pickle")
