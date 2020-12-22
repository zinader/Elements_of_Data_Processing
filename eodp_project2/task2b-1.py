import pandas as pd
import numpy as np
from numpy import nan
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectFromModel
import sklearn.metrics as sm
from pandas import merge
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn import datasets
import csv
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

missing_val = [".."]                
data2  = pd.read_csv('world.csv',na_values = missing_val)

data3 = data2.iloc[0:264,2]
data2 = data2.iloc[0:264,3:]
C = data2.values


    
imput = SimpleImputer(missing_values=np.nan,strategy = "median")
imput = imput.fit(C[:,:])
C[:,:] = imput.transform(C[:,:])

#Getting 210 features using Interaction Pairs

ds = pd.DataFrame(C[:,:])
dv = ds

frames = [data3,dv]
with_code = pd.concat(frames,axis=1, sort=False)
data4  = pd.read_csv('life.csv')
op = merge(with_code,data4,on = "Country Code")
op = op.drop(columns=['Country','Year','Country Code'], axis=1)
#print(op)

interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_inter = interaction.fit_transform(ds)
dp = pd.DataFrame(X_inter)

#Getting the last 1 feature using K-Means Clustering
#Scaling the dataframe before clustering
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(dp))


list4 = []
list5 = []
list6 = []
#Getting the last 1 feature using K-Means Clustering
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_normalized)
    y_kmeans = kmeans.predict(df_normalized.iloc[:,:])


    #Concatenating the last k means feature
    dc = pd.DataFrame(y_kmeans)


    frames = [df_normalized,dc]
    result = pd.concat(frames,axis=1, sort=False)
    result.columns = range(211)
    #print(result)
    second_result = result
    third_result = result
    
    


    # Merging the Data After Imputation and generating the features


    frames = [data3,result]
    with_code = pd.concat(frames,axis=1, sort=False)
    data4  = pd.read_csv('life.csv')
    da = merge(with_code,data4,on = "Country Code")
    da = da.drop(columns=['Country','Year'], axis=1)
    dt = da.iloc[:,1:-1]


    #Using Random Forest Classifier algorithm for feature engineering

    X = dt
    Y = da.iloc[:,-1]


    sel = SelectFromModel(RandomForestClassifier(n_estimators = 200,random_state = 100),max_features = 4)
    sel.fit(X, Y)
    feat_sel_df = dt.columns[(sel.get_support())]
    feat_sel_df = dt[feat_sel_df]
    X = feat_sel_df

    #print(X)
    dp = X
    classlabel=da.iloc[:,-1]

    X_train, X_test,y_train, y_test = train_test_split(dp,classlabel.values.ravel(), test_size=0.33, random_state=100)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)

    # Getting the accuracy score for the best 4 features using feature eng

    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred=knn.predict(X_test)
    #print("Accuracy of feature engineering: " + str(round(accuracy_score(y_test, y_pred),3)))
    list4.append(accuracy_score(y_test, y_pred))

    
    
    
    x = op.iloc[:,: -1].values
    # Separating out the target
    y = op.iloc[:,-1].values
    # Standardizing the features
    x = preprocessing.StandardScaler().fit_transform(x)

    #Using PCA algorithm for feature engineering

    from sklearn.decomposition import PCA
    pca = PCA(n_components=4)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['pc1', 'pc2','pc3','pc4'])

    #print(principalDf)
    # Getting the accuracy score for the best 4 features using PCA

    X_train, X_test,y_train, y_test = train_test_split(principalDf,classlabel.values.ravel(), test_size=0.33, random_state=100)
    
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    #print("Accuracy of PCA: " + str(round(accuracy_score(y_test, y_pred),3)))
    list5.append(accuracy_score(y_test, y_pred))
    # Getting the accuracy score for the FIRST 4 features 

    last_task = op.iloc[:,0:4]


    X_train, X_test,y_train, y_test = train_test_split(last_task,classlabel.values.ravel(), test_size=0.33, random_state=100)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    #print("Accuracy of first four features: " + str(round(accuracy_score(y_test, y_pred),3)))
    list4.append(accuracy_score(y_test, y_pred))
    
fig = sns.lineplot(x = list5,y = range(1,11))
plt.show()
plt.savefig('task2bgraph1.png')

kmeans = KMeans(n_clusters=6)
kmeans.fit(df_normalized)
y_kmeans = kmeans.predict(df_normalized.iloc[:,:])


#Concatenating the last k means feature
dc = pd.DataFrame(y_kmeans)


frames = [df_normalized,dc]
result = pd.concat(frames,axis=1, sort=False)
result.columns = range(211)
print(result)
second_result = result
third_result = result


# Merging the Data After Imputation and generating the features


frames = [data3,result]
with_code = pd.concat(frames,axis=1, sort=False)
data4  = pd.read_csv('life.csv')
da = merge(with_code,data4,on = "Country Code")
da = da.drop(columns=['Country','Year'], axis=1)
dt = da.iloc[:,1:-1]


#Using Random Forest Classifier algorithm for feature engineering

X = dt
Y = da.iloc[:,-1]


sel = SelectFromModel(RandomForestClassifier(n_estimators = 200,random_state = 100),max_features = 4)
sel.fit(X, Y)
feat_sel_df = dt.columns[(sel.get_support())]
feat_sel_df = dt[feat_sel_df]
X = feat_sel_df

print(X)
dp = X
classlabel=da.iloc[:,-1]

X_train, X_test,y_train, y_test = train_test_split(dp,classlabel.values.ravel(), test_size=0.33, random_state=100)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

# Getting the accuracy score for the best 4 features using feature eng

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)
print("Accuracy of feature engineering: " + str(round(accuracy_score(y_test, y_pred),3)))




x = op.iloc[:,: -1].values
# Separating out the target
y = op.iloc[:,-1].values
# Standardizing the features
x = preprocessing.StandardScaler().fit_transform(x)


#Using PCA algorithm for feature engineering

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2','pc3','pc4'])

print(principalDf)
# Getting the accuracy score for the best 4 features using PCA

X_train, X_test,y_train, y_test = train_test_split(principalDf,classlabel.values.ravel(), test_size=0.33, random_state=100)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
print("Accuracy of PCA: " + str(round(accuracy_score(y_test, y_pred),3)))

# Getting the accuracy score for the FIRST 4 features 

last_task = op.iloc[:,0:4]


X_train, X_test,y_train, y_test = train_test_split(last_task,classlabel.values.ravel(), test_size=0.33, random_state=100)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
print("Accuracy of first four features: " + str(round(accuracy_score(y_test, y_pred),3)))
    