import os
import pickle

import pandas as pd
import torch
import tqdm

class LoadData():
    def __init__(self, path="./Data/", dataset="frappe", loss_type="square_loss"):
        self.dataset = dataset
        self.loss_type = loss_type
        self.path = path + "/"
        self.trainfile = self.path + dataset + ".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".validation.libfm"
        self.features_M = {}
        self.construct_df()

    def construct_df(self):
        self.data_train = pd.read_table(self.trainfile, sep=" ", header=None, engine='python')
        self.data_test = pd.read_table(self.testfile, sep=" ", header=None, engine="python")
        self.data_valid = pd.read_table(self.validationfile, sep=" ", header=None, engine="python")

        for i in self.data_test.columns[1:]:
            self.data_test[i] = self.data_test[i].apply(lambda x: int(x.split(":")[0]))
            self.data_train[i] = self.data_train[i].apply(lambda x: int(x.split(":")[0]))
            self.data_valid[i] = self.data_valid[i].apply(lambda x: int(x.split(":")[0]))

        self.all_data = pd.concat([self.data_train, self.data_test, self.data_valid])
        self.field_dims = []

        for i in self.all_data.columns[1:]:
            maps = {val: k for k, val in enumerate(set(self.all_data[i]))}
            self.data_test[i] = self.data_test[i].map(maps)
            self.data_train[i] = self.data_train[i].map(maps)
            self.data_valid[i] = self.data_valid[i].map(maps)
            self.features_M[i] = maps
            self.field_dims.append(len(set(self.all_data[i])))
        self.data_test[0] = self.data_test[0].apply(lambda x: max(x, 0))
        self.data_train[0] = self.data_train[0].apply(lambda x: max(x, 0))
        self.data_valid[0] = self.data_valid[0].apply(lambda x: max(x, 0))


class RecData(torch.utils.data.Dataset):
    def __init__(self, all_data):
        self.data_df = all_data

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        x = self.data_df.iloc[idx].values[1:]
        y1 = self.data_df.iloc[idx].values[0]
        return x, y1

def getdataloader_frappe(path="", dataset="frappe", num_ng=4, batch_size=256):
    # dataset spilt:7 2 1 
    print(os.getcwd())
    print("start loading data")
    path = path
    path_ml = path +'preprocess-frappe.p'
    if not os.path.exists(path_ml):
        DataF = LoadData(path=path, dataset=dataset)
        pickle.dump((DataF.data_test, DataF.data_train, DataF.data_valid, DataF.field_dims), open(path_ml, 'wb'))
        print("data process and save success")
    else:
        print("data exist and load it directly")
    data_test, data_train, data_valid, field_dims = pickle.load(open(path_ml, mode='rb'))

    datatest = RecData(data_test)
    datatrain = RecData(data_train)
    datavalid = RecData(data_valid)
    print("datatest", len(datatest))
    print("datatrain", len(datatrain))
    print("datavalid", len(datavalid))
    trainLoader = torch.utils.data.DataLoader(datatrain, batch_size=batch_size, shuffle=True, num_workers=2,
                                              pin_memory=True, drop_last=True)
    validLoader = torch.utils.data.DataLoader(datavalid, batch_size=batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)
    testLoader = torch.utils.data.DataLoader(datatest, batch_size=batch_size, shuffle=False, num_workers=2,
                                             pin_memory=True)
    return field_dims, trainLoader, validLoader, testLoader


if __name__ == '__main__':
    import os 
    print(os.getcwd())
    field_dims, trainLoader, validLoader, testLoader = getdataloader_frappe(path="", batch_size=256)
    for _ in tqdm.tqdm(trainLoader):
        pass
    it = iter(trainLoader)
    print(next(it)[0])
    print(field_dims)
