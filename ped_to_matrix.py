
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.base import BaseEstimator, TransformerMixin

def base_from_ped(chr_merge):

    seq_all=[]
    for i in range(chr_merge.shape[0]):
        seq = chr_merge.iloc[i,6:].values
        seq_all.append(seq)
    return(pd.DataFrame(seq_all))


def base_trans_num(seq_all_df):

    trans_code = {"AA":0,"AT":1,"TA":1,"AC":2,"CA":2,"AG":3,"GA":3,"TT":4,"TC":5,"CT":5,"TG":6,"GT":6,"CC":7,"CG":8,"GC":8,"GG":9}
    code_arr = []
    for row in range(seq_all_df.shape[0]):
        code_list = []
        for base in range(0,seq_all_df.shape[1],2):
            joint_seq = seq_all_df.iloc[row,base]+seq_all_df.iloc[row,base+1]
            num_code = trans_code[joint_seq]
            code_list.append(num_code)
        code_arr.append(code_list)
    code_df = pd.DataFrame(code_arr)
    return(code_df)


class MICSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=10000):

        self.k = k
        self.mic_scores_ = None
        self.top_k_indices_ = None

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y).ravel()
        self.mic_scores_ = mutual_info_regression(X, y)
        self.top_k_indices_ = np.argsort(self.mic_scores_)[-self.k:]
        return self

    def transform(self, X):

        X = np.array(X)
        return X[:, self.top_k_indices_]


def split_and_select_features_sklearn(data, label, num_features=10000, test_size=0.2, random_state=42):


    """

    """
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size, random_state=random_state)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    mic_selector = MICSelector(k=num_features)

    mic_selector.fit(X_train, y_train)
    X_train_selected = mic_selector.transform(X_train)
    X_test_selected = mic_selector.transform(X_test)

    return X_train_selected, X_test_selected, y_train, y_test

if __name__=="__main__":

    ped = pd.read_csv("./test.ped", sep=" ", header=None)
    label = pd.read_csv("./test_label.csv")


    seq_all_df = base_from_ped(ped)
    code_df = base_trans_num(seq_all_df)
    data = pd.DataFrame(code_df)

    X_train_selected, X_test_selected, y_train, y_test = split_and_select_features_sklearn(data, label,
                                                                    num_features=10000)

    pd.DataFrame(X_train_selected).to_csv("./X_train.csv", index=None)
    pd.DataFrame(X_test_selected).to_csv("./X_test.csv", index=None)
    pd.DataFrame(y_train).to_csv("./y_train.csv", index=None)
    pd.DataFrame(y_test).to_csv("./y_test.csv", index=None)
    
    print("successfully")