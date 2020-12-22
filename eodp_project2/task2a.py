import pandas as pd
import numpy as np
from numpy import nan
from sklearn.impute import SimpleImputer
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from pandas import merge
import csv


with open('world.csv') as f:
    iris=[line for line in csv.reader(f)]
    
with open('life.csv') as f:
    iris2=[line for line in csv.reader(f)]
    

missing_val = [".."]                
data3  = pd.read_csv('world.csv',na_values = missing_val)
data4  = pd.read_csv('life.csv')
    

X = data3[0:264]
B = X.iloc[:,0:3].values
C = X.iloc[:,3:].values

#IMPUTATION  

imput = SimpleImputer(missing_values=np.nan,strategy = "median")
imput = imput.fit(C[:,:])
C[:,:] = imput.transform(C[:,:])


dd = pd.DataFrame(C[:,:]) 
dc = pd.DataFrame(B[:,:]) 

c = 0    
i = 0
k = 3

with open('task2a.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(("feature","median","mean","variance"))
    while(i<20 and k<23):
        
        # MEDIAN
        # MEAN
        # VARIANCE
        
        median2 = dd.iloc[:, i].median()
        mean2 = dd.iloc[:, i].mean()
        var2 = dd.iloc[:, i].var()
        
        writer.writerow((iris[0][k],round(median2,3),round(mean2,3),round(var2,3)))
        dd.iloc[:, i] = dd.iloc[:, i] - dd.iloc[:, i].mean()
        dd.iloc[:, i] = dd.iloc[:, i]/dd.iloc[:, i].std()
        i = i+1;
        k = k+1;    
    

    
frames = [dc,dd]
result = pd.concat(frames,axis=1, sort=False)


str2 = ['Country Name', 'Time', 'Country Code', 'Access to electricity, rural (% of rural population) [EG.ELC.ACCS.RU.ZS]', 'Adjusted savings: particulate emission damage (% of GNI) [NY.ADJ.DPEM.GN.ZS]', 'Birth rate, crude (per 1,000 people) [SP.DYN.CBRT.IN]', 'Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total) [SH.DTH.COMM.ZS]', 'Cause of death, by non-communicable diseases (% of total) [SH.DTH.NCOM.ZS]', 'Domestic general government health expenditure per capita (current US$) [SH.XPD.GHED.PC.CD]', 'Individuals using the Internet (% of population) [IT.NET.USER.ZS]', 'Lifetime risk of maternal death (%) [SH.MMR.RISK.ZS]', 'Lifetime risk of maternal death (1 in: rate varies by country) [SH.MMR.RISK]', 'Maternal mortality ratio (modeled estimate, per 100,000 live births) [SH.STA.MMRT]', 'Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70, female (%) [SH.DYN.NCOM.FE.ZS]', 'Mortality rate attributed to household and ambient air pollution, age-standardized (per 100,000 population) [SH.STA.AIRP.P5]', 'Mortality rate attributed to household and ambient air pollution, age-standardized, female (per 100,000 female population) [SH.STA.AIRP.FE.P5]', 'Mortality rate attributed to household and ambient air pollution, age-standardized, male (per 100,000 male population) [SH.STA.AIRP.MA.P5]', 'Mortality rate attributed to unintentional poisoning, female (per 100,000 female population) [SH.STA.POIS.P5.FE]', 'Mortality rate attributed to unsafe water, unsafe sanitation and lack of hygiene (per 100,000 population) [SH.STA.WASH.P5]', 'People using at least basic drinking water services (% of population) [SH.H2O.BASW.ZS]', 'People using at least basic sanitation services (% of population) [SH.STA.BASS.ZS]', 'People using at least basic sanitation services, urban (% of urban population) [SH.STA.BASS.UR.ZS]', 'Prevalence of anemia among children (% of children under 5) [SH.ANM.CHLD.ZS]']

result.columns = str2

#MERGING life.csv and world.csv after imputing world.csv

da = merge(result,data4,on = "Country Code")
da = da.drop(columns=['Country','Year'], axis=1)

data5  = da

Z = data5.iloc[:,3:-1].values

ds = pd.DataFrame(Z[:,:])

classlabel=data5.iloc[:,-1]

        
X_train, X_test,y_train, y_test = train_test_split(ds,classlabel.values.ravel(), test_size=0.33, random_state=100)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#Getting the Accuracy for Prediction

dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, y_train)
y_pred=dt.predict(X_test)
print("Accuracy of decision tree: " + str(round(accuracy_score(y_test, y_pred),3)))
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
print("Accuracy of k-nn (k=5): " + str(round(accuracy_score(y_test, y_pred),3)))
knp = neighbors.KNeighborsClassifier(n_neighbors=10)
knp.fit(X_train, y_train)
y_pred=knp.predict(X_test)
print("Accuracy of k-nn (k=10): " + str(round(accuracy_score(y_test, y_pred),3)))


