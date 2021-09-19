import torch
import math
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, RobustScaler


SEED = 0


def split_data(X, Y, percentage):
    num_val = int(len(X)*percentage)
    return X[num_val:], Y[num_val:], X[:num_val], Y[:num_val]


def split_validation_test(X, Y, p1=0.4, p2=0.5):
    X, Y, Xval, Yval = split_data(X, Y, p1)  # p1's default is 0.4 so 60% of the data is training data.
    Xval, Yval, Xtest, Ytest = split_data(Xval, Yval, p2)  # p2's default is 0.5 so 20% is validation and 20% is test.
    return X, Y, Xval, Yval, Xtest, Ytest


def shuffle(X, Y):
    X_dim = len(X[0])
    data = torch.cat((X, Y.view(-1, 1)), 1)
    data = data[torch.randperm(data.size()[0])]
    X = data[:, :X_dim]
    Y = data[:, X_dim]
    return X, Y


def gen_sklearn_data(x_dim, n_samples, informative_frac=1, shift_range=1, scale_range=1, noise_frac=0.01):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_informative = int(informative_frac*x_dim)
    n_redundant = x_dim - n_informative
    shift_arr = shift_range*np.random.randn(x_dim)
    scale_arr = scale_range*np.random.randn(x_dim)
    X, Y = make_classification(n_samples=n_samples, n_features=x_dim, n_informative=n_informative, n_redundant=n_redundant,
                               flip_y=noise_frac, shift=shift_arr, scale=scale_arr, random_state=0)
    Y[Y == 0] = -1
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return torch.from_numpy(X), torch.from_numpy(Y)


def gen_custom_normal_data(x_dim, n_samples, pos_mean, pos_std, neg_mean, neg_std):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    pos_samples_num = n_samples // 2
    neg_samples_num = n_samples - pos_samples_num
    posX = torch.randn((pos_samples_num, x_dim)) * pos_std + pos_mean
    negX = torch.randn((neg_samples_num, x_dim)) * neg_std + neg_mean

    X = torch.cat((posX, negX), 0)
    Y = torch.unsqueeze(torch.cat((torch.ones(len(posX)), -torch.ones(len(negX))), 0), 1)

    X, Y = shuffle(X, Y)
    return X, Y


def load_spam_data():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    path = r".\Datasets\IS_journal_tip_spam.arff"
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    most_disc = ['qTips_plc', 'rating_plc', 'qEmail_tip', 'qContacts_tip', 'qURL_tip', 'qPhone_tip', 'qNumeriChar_tip', 'sentistrength_tip', 'combined_tip', 'qWords_tip', 'followers_followees_gph', 'qunigram_avg_tip', 'qTips_usr', 'indeg_gph', 'qCapitalChar_tip', 'class1']
    df = df[most_disc]
    df["class1"].replace({b'spam': -1, b'notspam': 1}, inplace=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    Y = df['class1'].values
    X = df.drop('class1', axis=1).values
    x_dim = len(X[0])
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    X /= math.sqrt(x_dim)
    return torch.from_numpy(X), torch.from_numpy(Y)


def load_card_fraud_data():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df = pd.read_csv('./Datasets/creditcard.csv')

    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    scaled_amount = df['scaled_amount']
    df.drop(['scaled_amount'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)

    df["Class"].replace({1: -1, 0: 1}, inplace=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # amount of fraud classes 492 rows.
    fraud_df = df.loc[df['Class'] == -1]
    non_fraud_df = df.loc[df['Class'] == 1][:492]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    # Shuffle dataframe rows
    df = normal_distributed_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    Y = df['Class'].values
    X = df.drop('Class', axis=1).values
    x_dim = len(X[0])
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    X /= math.sqrt(x_dim)
    
    return torch.from_numpy(X), torch.from_numpy(Y)


def load_credit_default_data():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    url = 'https://raw.githubusercontent.com/ustunb/actionable-recourse/master/examples/paper/data/credit_processed.csv'
    df = pd.read_csv(url)
    df["NoDefaultNextMonth"].replace({0: -1}, inplace=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    df = df.drop(['Married', 'Single', 'Age_lt_25', 'Age_in_25_to_40', 'Age_in_40_to_59', 'Age_geq_60'], axis=1)

    fraud_df = df.loc[df["NoDefaultNextMonth"] == -1]
    non_fraud_df = df.loc[df["NoDefaultNextMonth"] == 1][:6636]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    # Shuffle dataframe rows
    df = normal_distributed_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    scaler = StandardScaler()
    df.loc[:, df.columns != "NoDefaultNextMonth"] = scaler.fit_transform(df.drop("NoDefaultNextMonth", axis=1))
    Y, X = df.iloc[:, 0].values, df.iloc[:, 1:].values
    x_dim = len(X[0])
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    X /= math.sqrt(x_dim)
    return torch.from_numpy(X), torch.from_numpy(Y)


def load_financial_distress_data():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data = pd.read_csv("./Datasets/Financial Distress.csv")

    data = data[data.columns.drop(list(data.filter(regex='x80')))]  # Since it is a categorical feature with 37 features.
    data.drop(['Time'], axis=1, inplace=True)

    data_grouped = data.groupby(['Company']).last()

    scaler = StandardScaler()
    data_grouped.loc[:, data_grouped.columns != "Financial Distress"] = scaler.fit_transform(data_grouped.drop("Financial Distress", axis=1))

    # Shuffle dataframe rows
    data_grouped = data_grouped.sample(frac=1, random_state=SEED).reset_index(drop=True)

    Y, X = data_grouped.iloc[:, 0].values, data_grouped.iloc[:, 1:].values
    for y in range(0, len(Y)):  # Converting target variable from continuous to binary form
        if Y[y] < -0.5:
            Y[y] = -1
        else:
            Y[y] = 1
    x_dim = len(X[0])
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    X /= math.sqrt(x_dim)
    return torch.from_numpy(X), torch.from_numpy(Y)
