import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from czech_holidays import czech_holidays
import datetime


class DF:

    def create_df(self, data, i):
        '''
        Creates DataFrame for a specific bulding denoted by index i
        '''
        df_name = list(data.keys())[i]
        df = pd.DataFrame(list(data.values())[i], columns=['occ', 'temp', 'meas'])
        
        return df, df_name



    def find_least_most_missing_df(self, data):
        '''
        Finds the building with the least and the most missing percentage of values from the whole dataset "data"
        '''
        least_percentage = 100
        most_percentage = 0
        for i in range(len(data)):
            df = pd.DataFrame(list(data.values())[i], columns=['occ', 'temp', 'meas'])
            length = len(df)
            missing = len(df) - len(df.dropna())
            percentage = missing / length * 100

            if least_percentage > percentage:
                least_percentage = percentage
                least_missing = missing
                least_index = i
            
            if most_percentage < percentage:
                most_percentage = percentage
                most_missing = missing
                most_index = i

        print(f'File "{list(data.keys())[most_index]}.csv" with lenght {len(list(data.values())[most_index])} has the most rows that are missing some values: {most_missing} ({most_percentage:.3f}% of data is missing)')
        print(f'File "{list(data.keys())[least_index]}.csv" with lenght {len(list(data.values())[least_index])} has the least rows that are missing some values: {least_missing} ({least_percentage:.3f}% of data is missing)')
        
        return least_index, most_index



    def add_day_time(self, df):
        '''
        Based on the index that is a timestamp, this function takes as an input DataFrame "df"
        and add following columns to the DataFrame:
                'hour' 
                'minute' 
                'dayofweek'
                'quarter'
                'month'
                'year'
                'dayofyear'
                'dayofmonth'
                'weekofyear'
        '''
        df['datetime'] = df.index
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['dayofweek'] = df['datetime'].dt.isocalendar().day.astype(np.int64) #1-7, needs to be converted from uint32 to int64 so it can be fed to the model
        # df['dayofweek'] = df['datetime'].dt.dayofweek #0-6
        df['quarter'] = df['datetime'].dt.quarter
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['dayofyear'] = df['datetime'].dt.dayofyear
        df['dayofmonth'] = df['datetime'].dt.day
        df['weekofyear'] = df['datetime'].dt.isocalendar().week.astype(np.int64) #needs to be converted from uint32 to int64 so it can be fed to the model

        return df



    def add_features(self, df):
        '''
        Adding 2 features that can be used for analyzing predictions (e.g. anomaly detection), not for future predictions:
            - adding average meas for the month as a new feature
            - adding average meas for a dayofweek in a specific month as a new feature
        '''
        df['meas_month_avg'] = df.groupby(['month', 'year'])['meas'].transform('mean') #calculating average meas for a specific month
        df['meas_dayofweek_avg'] = df.groupby(['dayofweek', 'month', 'year'])['meas'].transform('mean') #calculating average meas for a dayofweek in a specific month
        #based on the test dfs either keep only 'meas_dayofweek_avg' or both 'meas_dayofweek_avg' and 'meas_month_avg' (can be tested in data exploration)

        return df

    
    
    def add_lags(self, df):
        '''
        Adding lag features - what was the target variable 'meas' (x) days in the past and that information is fed to the model as a new feature:
            - adding meas value year ago
            - adding meas value week ago
        '''
        target_map = df['meas'].to_dict()
        df['lag_1year'] = (df.index - pd.Timedelta('364 days')).map(target_map)  #364/7=52 so it'll give us exactly the same day of the week
        df['lag_1week'] = (df.index - pd.Timedelta('6 days')).map(target_map)

        return df



    def print_info(self, df, df_name):
        '''
        Prints info about the DataFrame 
                - the df itself
                - name 
                - lenght 
                - shape 
                - how many occupied and unoccupied values there are
                - the sum of missing values 
                - df.describe
        '''
        print(f'\nInfo about DataFrame from file {df_name}.csv')
        print(f'DataFrame: \n{df}')
        print(f'\nIndices: \n{df.index}')
        print(f'\nShape of DF is: {df.shape}')
        print(f'\nOccupancy summary: \n{df["occ"].value_counts()}')
        print(f'\nDescription of DF: \n {df.describe()}')
        print(f'\nThe dataset has total of {len(df)} rows. \nSum of missing values for each column is: \n{df.isnull().sum()}\n')



    def find_duplicates(self, df):
        '''
        Identifies and prints duplicated rows in the dataset
        '''
        print(df.duplicated().value_counts())
        for i in range(len(df)):
            if df.duplicated()[i] == True:
                print(df.iloc[i, 0:])



    def create_holiday_df(self):
        '''
        Creates a DataFrame with all czech public holidays from the year 2018 to 2022
        '''
        holidays2018 = pd.DataFrame(czech_holidays(2018), columns=['hol_date', 'hol_name_cz', 'hol_name_en']).set_index('hol_date', drop=False)
        holidays2019 = pd.DataFrame(czech_holidays(2019), columns=['hol_date', 'hol_name_cz', 'hol_name_en']).set_index('hol_date', drop=False)
        holidays2020 = pd.DataFrame(czech_holidays(2020), columns=['hol_date', 'hol_name_cz', 'hol_name_en']).set_index('hol_date', drop=False)
        holidays2021 = pd.DataFrame(czech_holidays(2021), columns=['hol_date', 'hol_name_cz', 'hol_name_en']).set_index('hol_date', drop=False)
        holidays2022 = pd.DataFrame(czech_holidays(2022), columns=['hol_date', 'hol_name_cz', 'hol_name_en']).set_index('hol_date', drop=False)

        holidays_list = [holidays2018, holidays2019, holidays2020, holidays2021, holidays2022]
        holidays_df = pd.concat(holidays_list)

        holidays_df.index = pd.to_datetime(holidays_df.index)

        return holidays_df



    def next_weekday(self, d, weekday):
        '''
        Finds the the next specified day (Monday-Sunday) after a certain date
        '''
        days_ahead = weekday - d.weekday()

        if days_ahead <= 0: #Target day already happened this week
            days_ahead += 7

        return d + datetime.timedelta(days_ahead)




class Figure:

    def plot_fig(self, df, df_name):
        '''
        Plots temperature and meas for the specified DataFrame
        '''
        fig, axs = plt.subplots(2, figsize=(10, 10), sharex=True, sharey=False)
        fig.suptitle(f'Data from file "{df_name}"!')
        plt.xticks(rotation = 45)

        axs[0].plot(df['temp'])
        axs[0].set_ylabel('temp')
        axs[1].plot(df['meas'])
        axs[1].set_ylabel('meas')

        plt.show()



    def plot_heatmap(self, df):
        '''
        Plots heatmap for the specified DataFrame which shows the correlated features (Pearson correlation coefficients)
        --------------------------------
        Pearson Correlation Coefficient:
             - R = 1  -> Strong positive relationship
             - R = 0  -> Not linearly correlated
             - R = -1 -> Strong negative relationship
        '''
        colormap = plt.cm.RdBu
        plt.figure(figsize=(15, 10))
        sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=False, cmap=colormap, 
                    linecolor='white', annot=True, fmt='.2f') # getting either close to 1 or -1 means some sort of strong linear correlation
        plt.title('Pearson Correlation of Features')
        plt.xticks(rotation = 10)
        plt.yticks(rotation = 45)
        plt.tight_layout()

        plt.show()




class Modifier:

    def drop_missing_val(self, df):
        '''
        Drops all rows with missing values from a DataFrame
        '''
        df_no_NaN = df.dropna()
        print(f'\n{len(df)-len(df_no_NaN)} rows were dropped ouf of {len(df)} rows.')

        return df_no_NaN



    #Buildings with only one value for occupancy: 1 - True, 5 - True, 6 - True, 7 - True, 14 - True, 15 - True
    def encode_labels(self, df):
        '''
        LabelEncoder converts categorical variables in column 'occ' (True, False) to binary values (True -> 1, False -> 0)
        '''
        if len(df['occ'].value_counts().index) == 2:
            label_encoder = LabelEncoder()
            occ_cat = df['occ']
            occ_encoded = label_encoder.fit_transform(occ_cat) #'False' -> 0, 'True' -> 1 - does it alphabetically; thus, False=0, True=1
            occ_DF = pd.DataFrame(occ_encoded, columns=['occ'])
            df['occ'] = occ_DF['occ'].values #replacing values of 'True' with 1 and 'False' with 0 in column 'occ' 

        elif len(df['occ'].value_counts().index) == 1:  #when building has only all 'True' or 'False' values for occupancy
            label_encoder = LabelEncoder() 
            occ_cat = df['occ']

            if str(df['occ'].value_counts().index[0]) == 'True':  #let's set LabelEncoder for buildings where there are only 'True' values
                occ_encoded = label_encoder.fit_transform(occ_cat) #'True' -> 0 (incorrect) => needs to be replaced by 1
                occ_DF = pd.DataFrame(occ_encoded, columns=['occ'])
                df['occ'] = occ_DF['occ'].values
                df['occ'].replace(to_replace = 0, value = 1, inplace=True) #replacing all 0 values with 1 (bc these all should be 'True')

            elif str(df['occ'].value_counts().index[0]) == 'False':     #let's set LabelEncoder for buildings where there are only 'False' values
                occ_encoded = label_encoder.fit_transform(occ_cat) #'False' -> 0 (correct) => no need for replacement
                occ_DF = pd.DataFrame(occ_encoded, columns=['occ'])
                df['occ'] = occ_DF['occ'].values

            else:
                print('LabelEncoder inside else')
        else:
            print('LabelEncoder outside else')

        return df



    def set_holiday_unoccupancy(self, df, holidays_df):
        '''
        When the building is not constantly in operation, the percentage of unoccupied days is calculated and 
        if it's more than 20% of the whole dataset, then the days that fall on czech public holidays are set to unoccupied ('occ' -> 0)
        '''
        pd.options.mode.chained_assignment = None
        if len(df['occ'].value_counts().index) == 2:
            percentage_unocc = df['occ'].value_counts().values[1] / (df['occ'].value_counts().values[0] + df['occ'].value_counts().values[1])
            if percentage_unocc > 0.2:  #if the percentace of unoccupied days is higher than 20%
                for row in range(len(df)):
                    if df.index[row] in holidays_df.index:  #if current index (datetime) is supposed to be holiday, set occupancy to 0
                        date_to_replace = df.index[row].date()
                        date_to_replace = "'" + str(date_to_replace) + "'"
                        df['occ'].loc[date_to_replace] = 0

        return df



    def scale(self, df):
        '''
        Using MinMaxScaler to scale values of 'temp' and 'meas' so models of buildings can be easily comparable. 
        Otherwise, it's not necessary for Gradient Boosting
        '''
        scaler = MinMaxScaler()
        minmax_temp = scaler.fit_transform(df[['temp']])
        minmax_meas = scaler.fit_transform(df[['meas']])
        minmax_temp_df = pd.DataFrame(minmax_temp, columns=['temp'], index=df.index)
        df['temp'] = minmax_temp_df['temp'].values
        minmax_temp_df = pd.DataFrame(minmax_meas, columns=['meas'], index=df.index)
        df['meas'] = minmax_temp_df['meas'].values

        return df



    def drop_outliers(self, df):
        '''
        Dropping outliers from the 'meas' feature based on Interquartile range (IQR)
        '''
        Q1 = df['meas'].quantile(0.25)
        Q3 = df['meas'].quantile(0.75)
        IQR = Q3 - Q1    #IQR is interquartile range

        filter = (df['meas'] >= Q1 - 1.5 * IQR) & (df['meas'] <= Q3 + 1.5 *IQR)
        df = df.loc[filter]

        return df




class Model:

    def mean_absolute_percentage_error(self, y_true, y_pred): 
        '''
        Calculates MAPE given y_true and y_pred
        '''
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



    def evaluation_metrics(self, df_eval, y_true, y_pred):
        '''
        Calculates the following evaluation metrics:
            - MEAN values
            - STD (Standard deviation)
            - MSE (Mean squared error)
            - MAE (Mean absolute error)
            - RMSE (Root mean squared error)
            - MAPE (Mean absolute percentage error)
            - R²
            - CV (Coefficient of Variation) = Relative Standard Deviation (RSD) = STD/MEAN
            - CV(RMSE) (Coefficient of Variation of the Root-Mean Squared Error) = RMSE/MEAN
        
        Args:
            y_true (float64): Y values for the dependent variable (test part), numpy array of floats 
            y_pred (float64): Predicted values for the dependent variable (test part), numpy array of floats
        
        Returns:
            MEAN, STD, MSE, MAE, RMSE, MAPE, R², CV, CV(RMSE) 
        '''    
        print('\nEvaluation metric results: ')
        print(f'\nMean values: \n{np.mean(df_eval, axis=0)}')
        print(f'\nStandar deviations: \n{np.std(df_eval)}')
        for i in range(len(df_eval.columns.values)):
            print(f'\tCoefficient of Variation (CV) for {df_eval.columns.values[i]}: {(np.std(df_eval[df_eval.columns.values[i]]) / np.mean(df_eval[df_eval.columns.values[i]], axis=0)):0.5f}')
        print(f'\n\tMean Squared Error (MSE): {mean_squared_error(y_true, y_pred):0.5f}')
        print(f'\tMean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):0.5f}')
        print(f'\tRoot Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_true, y_pred)):0.5f}')
        print(f'\tMean Absolute Percentage Error (MAPE): {Model().mean_absolute_percentage_error(y_true, y_pred):0.5f}')
        print(f'\tR2 score: {r2_score(y_true, y_pred):0.5f}')
        print(f'\tCoefficient of Variation of Root-Mean Squared Error (CV(RMSE)): {(np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(df_eval["meas"], axis=0)):0.5f}', end='\n\n')



    def write_evaluation_metrics(self, df_eval, y_true, y_pred):
        '''
        Calculates the following evaluation metrics:
            - MEAN values
            - STD (Standard deviation)
            - MSE (Mean squared error)
            - MAE (Mean absolute error)
            - RMSE (Root mean squared error)
            - MAPE (Mean absolute percentage error)
            - R²
            - CV (Coefficient of Variation) = Relative Standard Deviation (RSD) = STD/MEAN
            - CV(RMSE) (Coefficient of Variation of the Root-Mean Squared Error) = RMSE/MEAN
        
        Args:
            y_true (float64): Y values for the dependent variable (test part), numpy array of floats 
            y_pred (float64): Predicted values for the dependent variable (test part), numpy array of floats
        
        Write into file:
            MEAN, STD, MSE, MAE, RMSE, MAPE, R², CV, CV(RMSE)  
        ''' 
        with open("data_exploration/TOWT_vs_XGB.txt", "a") as file:   
            file.write('\nEvaluation metric results: ')
            file.write(f'\nMean values: \n{np.mean(df_eval, axis=0)}')
            file.write(f'\nStandar deviations: \n{np.std(df_eval)}')
            for i in range(len(df_eval.columns.values)):
                file.write(f'\n\tCoefficient of Variation (CV) for {df_eval.columns.values[i]}: {(np.std(df_eval[df_eval.columns.values[i]]) / np.mean(df_eval[df_eval.columns.values[i]], axis=0)):0.5f}')
            file.write(f'\n\tMean Squared Error (MSE): {mean_squared_error(y_true, y_pred):0.5f}')
            file.write(f'\n\tMean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):0.5f}')
            file.write(f'\n\tRoot Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_true, y_pred)):0.5f}')
            file.write(f'\n\tR2 score: {r2_score(y_true, y_pred):0.5f}')
            file.write(f'\n\tCoefficient of Variation of Root-Mean Squared Error (CV(RMSE)): {(np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(df_eval["meas"], axis=0)):0.5f}\n\n')