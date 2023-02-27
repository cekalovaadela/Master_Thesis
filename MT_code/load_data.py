from os import listdir
from os.path import isfile, join
import pickle
import pandas as pd

class Merger:
    def merge_files(self):
        path = '.\\csvExportDiplomka'

        #Find all files in dir 'path'
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

        data = {}

        #Read data from all files
        for f in onlyfiles:
            source, var = f[:-4].split('-')
            print(f'Reading {f}...')
            data[f'{source}_{var}'] = pd.read_csv(f'.\\csvExportDiplomka\\{f}', sep=';', dtype={'occ': 'boolean', 'temp': float, 'meas': float}, parse_dates=['ts'], decimal=",").set_index('ts')

        #Save data to python pickle file
        filename = '.\\data\\data.pickle'
        with open(filename, 'bw') as file:
            pickle.dump(data, file)
            print(filename, "was created.", sep=" ")
