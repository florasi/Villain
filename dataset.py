import itertools
import warnings
from datetime import datetime
from uuid import uuid4

import imblearn
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

warnings.filterwarnings("ignore")

dataset_path = "data/"


def split_data(dataset, worker_list=None, n_workers=2):
    if worker_list is None:
        worker_list = list(range(0, n_workers))
    # counter to create the index of different data samples
    idx = 0
    # dictionary to accomodate the split data
    dic_single_datasets = {}
    for worker in worker_list:
        dic_single_datasets[worker] = []
    """
    Loop through the dataset to split the data and labels vertically across workers.
    Splitting method from @abbas5253: https://github.com/abbas5253/SplitNN-for-Vertically-Partitioned-Data/blob/master/distribute_data.py
    """
    label_list = []
    index_list = []
    index_list_UUID = []
    height = dataset[0][0].shape[-1] // len(worker_list)
    for tensor, label in dataset:
        tensor = tensor.unsqueeze(0)
        i = 0
        uuid_idx = uuid4()
        for worker in worker_list[:-1]:
            dic_single_datasets[worker].append(tensor[:, :, :, height *
                                                      i:height * (i + 1)])
            i += 1

        dic_single_datasets[worker_list[-1]].append(tensor[:, :, :,
                                                           height * (i):])
        label_list.append(torch.Tensor([label]))
        index_list_UUID.append(uuid_idx)
        index_list.append(torch.Tensor([idx]))
        idx += 1

    return dic_single_datasets, label_list, index_list, index_list_UUID


def load_mnist(n_workers, dataset_path):
    transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
    train_set = datasets.MNIST(dataset_path,
                               train=True,
                               download=True,
                               transform=transform)
    val_set = datasets.MNIST(dataset_path,
                             train=False,
                             download=True,
                             transform=transform)
    img_workers, _, _, _ = split_data(train_set,n_workers=n_workers)
    img_workers_val, _, _, _, = split_data(val_set,n_workers=n_workers)
    split_train_dataset = []
    split_val_dataset = []
    for i in range(n_workers):
        split_train_dataset.append(torch.cat(img_workers[i]))
        split_val_dataset.append(torch.cat(img_workers_val[i]))
    train_labels = train_set.targets[:]
    val_labels = val_set.targets[:]
    return split_train_dataset, split_val_dataset, train_labels, val_labels


def load_cifar10(n_workers, dataset_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = datasets.CIFAR10(dataset_path,
                                 train=True,
                                 download=True,
                                 transform=transform)

    val_set = datasets.CIFAR10(dataset_path,
                               train=False,
                               download=True,
                               transform=transform)
    img_workers, _, _, _ = split_data(train_set,n_workers=n_workers)
    img_workers_val, _, _, _, = split_data(val_set,n_workers=n_workers)
    split_train_dataset = []
    split_val_dataset = []
    for i in range(n_workers):
        split_train_dataset.append(torch.cat(img_workers[i]))
        split_val_dataset.append(torch.cat(img_workers_val[i]))
    train_labels = train_set.targets[:]
    val_labels = val_set.targets[:]
    train_labels = torch.Tensor(train_labels)
    val_labels = torch.Tensor(val_labels)
    return split_train_dataset, split_val_dataset, train_labels, val_labels


def load_imagenette(n_workers, dataset_path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = datasets.ImageFolder(dataset_path + "imagenette2-160/train",
                                     transform=transform)

    val_set = datasets.ImageFolder(dataset_path + "imagenette2-160/val",
                                   transform=transform)
    img_workers, _, _, _ = split_data(train_set,n_workers=n_workers)
    img_workers_val, _, _, _, = split_data(val_set,n_workers=n_workers)
    split_train_dataset = []
    split_val_dataset = []
    for i in range(n_workers):
        split_train_dataset.append(torch.cat(img_workers[i]))
        split_val_dataset.append(torch.cat(img_workers_val[i]))
    train_labels = train_set.targets[:]
    val_labels = val_set.targets[:]
    train_labels = torch.Tensor(train_labels)
    val_labels = torch.Tensor(val_labels)
    return split_train_dataset, split_val_dataset, train_labels, val_labels


def load_cinic10(n_workers, dataset_path):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                             [0.24205776, 0.23828046, 0.25874835]),
    ])

    train_set = datasets.ImageFolder(dataset_path + "CINIC10/train",
                                     transform=transform)

    val_set = datasets.ImageFolder(dataset_path + "CINIC10/valid",
                                   transform=transform)
    img_workers, _, _, _ = split_data(train_set,n_workers=n_workers)
    img_workers_val, _, _, _, = split_data(val_set,n_workers=n_workers)
    split_train_dataset = []
    split_val_dataset = []
    for i in range(n_workers):
        split_train_dataset.append(torch.cat(img_workers[i]))
        split_val_dataset.append(torch.cat(img_workers_val[i]))
    train_labels = train_set.targets[:]
    val_labels = val_set.targets[:]
    train_labels = torch.Tensor(train_labels)
    val_labels = torch.Tensor(val_labels)
    return split_train_dataset, split_val_dataset, train_labels, val_labels



#################################Give me some credit######################################
# 用均值填充NaN
def mean_replace(df, columnName):
    df_mean = df.copy()
    impute_mean = SimpleImputer()
    raw_data = df_mean[columnName].values.reshape(-1, 1)
    new_data = impute_mean.fit_transform(raw_data)
    df_mean[columnName] = new_data
    return df_mean


# 用0填充NaN
def zero_replace(df, columnName):
    # 用0填充年龄的缺失值
    df0 = df.copy()  # 复制原数据，避免原数据被覆盖
    # 实例化
    impute_0 = SimpleImputer(strategy='constant', fill_value=0)
    # 去除Age属性的原始数据，并通过values转化为一维数组，在通过reshape变为二维数组
    # 因为sklearn中传到数据必须是二维的
    raw_data = df0[columnName].values.reshape(-1, 1)
    # fit_transform()一步到位，返回填充后的数据
    new_data = impute_0.fit_transform(raw_data)
    # 用新数据替换原数据
    df0[columnName] = new_data
    return df0


# 用0填充val == target
def zero_replace_sp(df, columnName, target):
    # 用0填充年龄的缺失值
    df0 = df.copy()  # 复制原数据，避免原数据被覆盖
    df0.loc[df0[columnName] == target, columnName] = 0
    return df0


# 用中位数处理NaN
def median_replace(df, columnName):
    df_median = df.copy()
    impute_median = SimpleImputer(strategy='median')
    raw_data = df_median[columnName].values.reshape(-1, 1)
    new_data = impute_median.fit_transform(raw_data)
    df_median[columnName] = new_data
    return df_median


def standardlize(df, columnName):
    data = df[columnName]
    return (data - data.mean()) / (data.std())


def manip_data(df):
    df = df[~df["age"].isin([0])]
    # 用中位数填充月收入
    df = median_replace(df, "MonthlyIncome")
    # 用0填充家人
    df = zero_replace(df, "NumberOfDependents")
    # 用0填充欠款逾期次数
    #  df = df[~df["NumberOfTimes90DaysLate"].isin([96, 98])]
    for col in ("NumberOfTime30-59DaysPastDueNotWorse",
                "NumberOfTime60-89DaysPastDueNotWorse",
                "NumberOfTimes90DaysLate"):
        df = zero_replace_sp(df, col, 96)
        df = zero_replace_sp(df, col, 98)
    #     df.drop(df[np.isnan(df['MonthlyIncome'])].index, inplace=True)
    df["MonthlyIncome"] = df["MonthlyIncome"].apply(np.log1p)
    # log of RevolvingUtilizationOfUnsecuredLines
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"].apply(np.log1p)
    # 全部归一化
    for columnName in df.columns.tolist():
        if columnName in ("id", "SeriousDlqin2yrs"):
            continue
        df[columnName] = standardlize(df, columnName)
    # 增加YoungAge与OldAge两个特征
    df['YoungAge'] = [1 if x < 21 else 0 for x in df['age']]
    df['OldAge'] = [1 if x > 65 else 0 for x in df['age']]

    return df


class CreditData(Dataset):

    def __init__(self, data, label):
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)

    def split_data(self,n_workers=2):
        k = int(10/n_workers)
        data = []
        for i in range(n_workers):
            data.append([])
        labels = []
        for d, l in self:
            for i in range(n_workers):
                data[i].append(d[i*k:k*(i+1)])
            labels.append(l)
        result = []
        for i in range(n_workers):
            result.append(torch.cat(data[i],dim=0).reshape(-1,k))
        return result, labels


def load_credit(n_workers,dataset_path):
    data = pd.read_csv(dataset_path + "GiveMeSomeCredit/cs-training.csv")
    data.columns = ["id"] + data.columns.tolist()[1:]
    data = manip_data(data)
    selectedColumns = [
        "RevolvingUtilizationOfUnsecuredLines", "age",
        "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
        "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
        "NumberOfOpenCreditLinesAndLoans", "MonthlyIncome",
        "NumberRealEstateLoansOrLines", "NumberOfDependents"
    ]
    smote = SMOTE(random_state=404)
    X, y = smote.fit_resample(data[selectedColumns], data.SeriousDlqin2yrs)
    X = np.array(X)
    y = np.array(y)
    y = y.ravel()
    dataset = CreditData(X, y)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    dl, dl_local = CreditData.split_data(train_dataset,n_workers=n_workers)
    dl_val, dl_localval = CreditData.split_data(test_dataset,n_workers=n_workers)
    return dl,dl_val, dl_local, dl_localval



###############################Bank###################################

def feature_scaling(data, numeric_attrs):
    for i in numeric_attrs:
        std = data[i].std()
        if std != 0:
            data[i] = (data[i] - data[i].mean()) / std
        else:
            data = data.drop(i, axis=1)
    return data


def encode_cate_attrs(data, cate_attrs):
    data = encode_edu_attrs(data)
    cate_attrs.remove('education')
    for i in cate_attrs:
        dummies_df = pd.get_dummies(data[i])
        dummies_df = dummies_df.rename(columns=lambda x: i + '_' + str(x))
        data = pd.concat([data, dummies_df], axis=1)
        data = data.drop(i, axis=1)
    return data


def encode_bin_attrs(data, bin_attrs):
    for i in bin_attrs:
        data.loc[data[i] == 'no', i] = 0
        data.loc[data[i] == 'yes', i] = 1
    return data


def encode_edu_attrs(data):
    values = [
        "illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school",
        "professional.course", "university.degree"
    ]
    levels = range(1, len(values) + 1)
    dict_levels = dict(zip(values, levels))
    for v in values:
        data.loc[data['education'] == v, 'education'] = dict_levels[v]
    return data


def trans_num_attrs(data, numeric_attrs):
    bining_num = 10
    bining_attr = 'age'
    data[bining_attr] = pd.qcut(data[bining_attr], bining_num)
    data[bining_attr] = pd.factorize(data[bining_attr])[0] + 1

    scaler = preprocessing.StandardScaler()
    for i in numeric_attrs:
        data[i] = scaler.fit_transform(data[i].values.reshape((-1, 1)))
    return data


def fill_unknown(data, bin_attrs, cate_attrs, numeric_attrs):
    # fill_attrs = ['education', 'default', 'housing', 'loan']
    fill_attrs = []
    for i in bin_attrs + cate_attrs:
        if data[data[i] == 'unknown']['y'].count() < 500:
            # delete col containing unknown
            data = data[data[i] != 'unknown']
        else:
            fill_attrs.append(i)

    data = encode_cate_attrs(data, cate_attrs)
    data = encode_bin_attrs(data, bin_attrs)
    data = trans_num_attrs(data, numeric_attrs)
    data['y'] = data['y'].map({'no': 0, 'yes': 1}).astype(int)
    data = data.replace({"unknown": np.nan})
    # data = data.interpolate(method='linear', limit_direction='forward', axis=0)
    data = data.replace({np.nan:0})
    return data


def train_predict_unknown(trainX, trainY, testX):
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainX, trainY)
    test_predictY = forest.predict(testX).astype(int)
    return pd.DataFrame(test_predictY, index=testX.index)


def preprocess_data_bank(dataset_path):
    input_data_path = dataset_path+"bank-additional/bank-additional-full.csv"
    data = pd.read_csv(input_data_path, sep=';')
    numeric_attrs = [
        'age',
        'duration',
        'campaign',
        'pdays',
        'previous',
        'emp.var.rate',
        'cons.price.idx',
        'cons.conf.idx',
        'euribor3m',
        'nr.employed',
    ]
    bin_attrs = ['default', 'housing', 'loan']
    cate_attrs = [
        'poutcome', 'education', 'job', 'marital', 'contact', 'month',
        'day_of_week'
    ]

    data = shuffle(data)
    data = fill_unknown(data, bin_attrs, cate_attrs, numeric_attrs)
    return data

class BankData(Dataset):

    def __init__(self, data, label):
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)

    def split_data(self,n_workers=2):
        k = int(48/n_workers)
        data = []
        for i in range(n_workers):
            data.append([])
        labels = []
        for d, l in self:
            for i in range(n_workers):
                data[i].append(d[i*k:k*(i+1)])
            labels.append(l)
        result = []
        for i in range(n_workers):
            result.append(torch.cat(data[i],dim=-1).reshape(-1,k))
        return result, labels


def load_bank(n_workers, dataset_path):
    data = preprocess_data_bank(dataset_path)
    data.columns = ["id"] + data.columns.tolist()[1:]
    smote=SMOTE(random_state=404)
    X,y=smote.fit_resample(data.drop(columns='y'),data.y)
    X = np.array(X)
    y = np.array(y)
    y = y.ravel()
    dataset = BankData(X, y)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    dl, dl_local = BankData.split_data(train_dataset,n_workers=n_workers)
    dl_val, dl_localval = BankData.split_data(test_dataset,n_workers=n_workers)
    return dl, dl_val, dl_local, dl_localval 


def load_gtsrb(n_workers, dataset_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                            (0.2724, 0.2608, 0.2669))
    ])
    train_set = datasets.GTSRB(dataset_path,
                                 split="train",
                                 download=True,
                                 transform=transform)

    val_set = datasets.GTSRB(dataset_path,
                               split="test",
                               download=True,
                               transform=transform)
    img_workers, _, _, _ = split_data(train_set,n_workers=n_workers)
    img_workers_val, _, _, _, = split_data(val_set,n_workers=n_workers)
    split_train_dataset = []
    split_val_dataset = []
    for i in range(n_workers):
        split_train_dataset.append(torch.cat(img_workers[i]))
        split_val_dataset.append(torch.cat(img_workers_val[i]))
    _, train_labels = zip(*train_set)
    _, val_labels = zip(*val_set)
    train_labels = torch.Tensor(train_labels)
    val_labels = torch.Tensor(val_labels)
    return split_train_dataset, split_val_dataset, train_labels, val_labels


def get_train_test_set(dataset, dataset_path=dataset_path,n_workers=2):
    if dataset == 'MNIST':
        return load_mnist(n_workers, dataset_path)
    elif dataset == 'cifar10' or dataset == 'Cifar10':
        return load_cifar10(n_workers, dataset_path)
    elif dataset == 'imagenette':
        return load_imagenette(n_workers, dataset_path)
    elif dataset == 'cinic10':
        return load_cinic10(n_workers, dataset_path)
    elif dataset == 'givemesomecredit':
        return load_credit(n_workers, dataset_path)
    elif dataset=="bank":
        return load_bank(n_workers, dataset_path)
    elif dataset=="gtsrb":
        return load_gtsrb(n_workers, dataset_path)
    else:
        raise KeyError(
            '{key} is not in supported datasets'.format(key=dataset))
