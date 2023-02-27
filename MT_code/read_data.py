import pickle


#Load data from pickle
class Reader:
    def read_data(self):
        with open('.\\data\\data.pickle', 'br') as file:
            data = pickle.load(file)
        print(f'{len(data)} files were loaded!\n')
        return data