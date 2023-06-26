<!-- omit in toc -->
# Predicting Economic Recession
<!-- omit in toc -->
## Table of Contents

- [Data Preparation](#data-preparation)
- [Dimensionality Reduction (PCA)](#dimensionality-reduction-pca)
- [Models](#models)
  - [Random Forest](#random-forest)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
- [Evaluation](#evaluation)

## Data Preparation

- [FRED Data Request](prep/1_fred_data_request.ipynb)
- [Comparing Recession Variables](prep/2_comparing_recession_variables.ipynb)
- [Data Filtering and Imputing](prep/3_data_filtering_and_imputing.ipynb)

## Dimensionality Reduction (PCA)

[View code](models/model_team14.py)

```
def select_features(metadata, X, threshold, criteria='cum'):
    
    n=len(metadata['group'].unique())-1

    for code in range(n):
        # separating a group of features which have the same code
        group = metadata[metadata['group']==code+1]
        # getting the names of these features from the metadata df
        group_feature_names = list(group['id'])
        # creating a subset of the full_data df having only features with the same code


        selected_cols=[colname for colname in group_feature_names if colname in X.columns]

        data = X[selected_cols]

        # Running feature scaling and PCA
        sc = StandardScaler()
        scaled = sc.fit_transform(data)
        # PCA
        pca = PCA()
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
```

## Models

[View code](models/1-1.pipeline_scaling.ipynb)

### Random Forest

```
clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=14), 
                       param_rf, cv=split, 
                       verbose=3, n_jobs=-1, scoring=['recall_macro'],
                    refit='recall_macro'
                    )
```

### K-Nearest Neighbors (KNN)

```
clf = GridSearchCV(KNeighborsClassifier(), param_knn, cv=split,  
                   verbose=3, n_jobs=-1, scoring=['recall_macro'],
                refit='recall_macro'
                )
```

### Support Vector Machine (SVM)

```
clf = GridSearchCV(SVC(random_state=14, class_weight=cw_dict, probability=True), 
                       param_svc, cv=split, 
                       verbose=3, n_jobs=-1, scoring=['recall_macro'],
                    refit='recall_macro'
                    )
```

## Evaluation

- [Model Selection](models/2.model_selection.ipynb)
- [Model Evaluation](models/3.evaluation.ipynb)
- [Sensitivity Analysis](models/4.sensitivity.ipynb)
