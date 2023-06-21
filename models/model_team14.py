import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import metrics

from scipy.spatial import distance
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import classification_report


def vis_tss(data, varname, n_split, test_size=None, gap=0):

    tss=TimeSeriesSplit(test_size=test_size, n_splits=n_split, gap=gap)

    fig, axs = plt.subplots(n_split, 1, figsize = (15,15), sharex = True) #Number of plots according to n_splits
    split = 0

    for train_index, test_index in tss.split(data):
        train_df = xy_data.iloc[train_index]
        test_df = xy_data.iloc[test_index]
        
        train_df[varname].plot(ax=axs[split],
                        label = 'Training Set',
                        title = f'Data Train/Test Split {split}')
        test_df[varname].plot(ax=axs[split],
                        label = 'Testing Set')
        axs[split].axvline(test_index.min(), color = 'black', ls='--')

        
        split +=1


        

def select_features(metadata, X, threshold, criteria='cum'):  ## criteria can be 'cum' or None (non-cumulative)
    
    n=len(metadata['group'].unique())-1

    for code in range(n):
        # # separating a group of features which have the same code
        group = metadata[metadata['group']==code+1]
        # # getting the names of these features from the metadata df
        group_feature_names = list(group['id'])
        # # creating a subset of the full_data df having only features with the same code


        selected_cols=[colname for colname in group_feature_names if colname in X.columns]

        data = X[selected_cols]

        # Running feature scaling and PCA
        sc = StandardScaler()
        scaled = sc.fit_transform(data)
        # PCA
        pca = PCA() # with no number of target PCs defined will not do any reduction
        pc = pca.fit_transform(scaled)
        n_pcs= pca.components_.shape[0]

        # getting the most important features and their names within this feature group
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
        
        # get the names
        most_important_names = [data.columns[most_important[i]] for i in range(n_pcs)]
        
        # Creating a dictionary to summarize features and importance into a dataframe
        dic = {'PC':['PC{}'.format(i) for i in range(n_pcs)],
            'variable':[most_important_names[i] for i in range(n_pcs)],
            'var_ratio':[pca.explained_variance_ratio_[i] for i in range(n_pcs)],
            'var_ratio_cum':np.cumsum([pca.explained_variance_ratio_[i] for i in range(n_pcs)]),
            'group':[code+1]*len(range(n_pcs))
        }
        
        if criteria=='cum':
            c_var='var_ratio_cum'
            threshold_idx=np.argwhere((dic[c_var]>threshold)*np.ones(n_pcs,dtype=int)==1).min()
            dic['select']=[1 if x<=threshold_idx else 0 for x in range(n_pcs)]
            
        else:
            c_var='var_ratio'
            threshold_idx=np.where(np.array(dic[c_var])>threshold)
            dic['select']=[1 if x in list(threshold_idx[0]) else 0 for x in range(n_pcs)]

        if code==0:
            df_most_important = pd.DataFrame(dic)
        
        else:
            df=pd.DataFrame(dic)
            df_most_important=pd.concat([df_most_important, df], axis=0, ignore_index=True)
    
    # build the dataframe
    df_most_important['title']=df_most_important['variable'].apply(lambda x:metadata[metadata.id==x]['title'].values[0])
    

    return df_most_important



def plot_pca(df_feature, metadata, k=3):

    n=len(metadata['group'].unique())-1

    if n%k==0:
        n_row=n//k
        n_col=k

    else:
        n_row=n//k+1
        n_col=k

    fig, ax=plt.subplots(nrows=n_row, ncols=n_col, squeeze=False, figsize=(5*n//k, 15))

    for code in range(n):

        if (code+1)%k==0:
            r=(code+1)//k-1
            c=n_col-1
        else:
            r=(code+1)//k
            c=(code+1)%k-1


        data=df_feature[df_feature.group==code+1]['var_ratio_cum'][:10]
        ax[r,c].plot(list(range(len(data))), data)
        ax[r,c].set_title('group{}'.format(code+1))
        ax[r,c].set_xlabel('number of components')
        ax[r,c].set_ylabel('explained var(cum)')
        plt.tight_layout();

        
# https://stackoverflow.com/questions/57015499/how-to-use-dynamic-time-warping-with-knn-in-python

# custom metric
def DTW(a, b):   
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost

    return cumdist[an, bn]


def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    
    for line in lines[2:len(lines)-5]:
        row = {}
        row_data = [val for val in line.split(' ') if val!='']
        row['class'] = round(float(row_data[0]),0)
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        row['accuracy']=float([val for val in lines[-4].split(' ') if val!=''][-2])
        report_data.append(row)
        
    df = pd.DataFrame(report_data)
    return df


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps ):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return torch.tensor(np.array(Xs)), torch.tensor(np.array(ys))


class RecModel_lstm(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size, num_layer, batch_first, dropout=0):
        super().__init__()
        self.n_features=n_features
        self.hidden_size=hidden_size
        self.num_layer=num_layer
        self.batch_first=batch_first
        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size, 
                            num_layers=num_layer, 
                            dropout=dropout,
                            batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, n_classes)
        
    def forward(self, x):
        out_x, (hidden, cell) = self.lstm(x)   ## for lstm
        out=hidden   ##[:,-1,:].reshape(-1)

        return self.linear(out)
    
    
class RecModel_gru(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size, num_layer, batch_first, dropout=0):
        super().__init__()
        self.n_features=n_features
        self.hidden_size=hidden_size
        self.num_layer=num_layer
        self.batch_first=batch_first
        self.gru = nn.GRU(input_size=n_features, 
                            hidden_size=hidden_size, 
                            num_layers=num_layer, 
                            dropout=dropout,
                            batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, n_classes)
        
    def forward(self, x):
        out_x, hidden = self.gru(x)   ## for lstm
        out=hidden   ##[:,-1,:].reshape(-1)
        
        return self.linear(out)
        
        
