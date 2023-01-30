%%time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif


import warnings
warnings.filterwarnings("ignore")

######################
#reading data 
df = pd.read_csv('creditcard.csv')

df.head()

X = df.drop(['Class'], axis=1)
y = df["Class"]
feature_list =  list(X.columns)







########Getting Feature list from the Logistic Regression (LR)



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

model_lr= SelectFromModel(LogisticRegression())
model_lr.fit(X_train, y_train)
selected_feat = X_train.columns[(model_lr.get_support())]
print ("LR Feature selection list ")
print (selected_feat)
lr_feature_sets=selected_feat


########Getting Feature list from the Kbest

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
 
    
#Kbest Feature Selection
model_kbest = SelectKBest()
model_kbest.fit(X,y)
model_kbest.get_support()
selected_feat= X_train.columns[(model_kbest.get_support())]
print ("KBEST Feature selection list ")
print (selected_feat)
kb_feature_sets=selected_feat



########Getting Feature list from the Random Forest ###################





# Get the best test size from the hyperparameter optimization results

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2

# Train a final model using the best hyperparameters
model_rf =SelectFromModel(RandomForestClassifier())
model_rf.fit(X_train, y_train)
model_rf.get_support()
selected_feat= X_train.columns[(model_rf.get_support())]
# print ("RF Feature selection list ")
# value_to_check = 'V9'
# if value_to_check in selected_feat:
#     print (selected_feat)
# #     selected_feat.drop('V9')
#     selected_feat.remove('V9')


print (selected_feat)
rf_feature_sets=selected_feat


###############################################################################





feature_pool =lr_feature_sets ;
feature_pool=feature_pool.append(kb_feature_sets)
feature_pool=feature_pool.append(rf_feature_sets)


feature_pool
