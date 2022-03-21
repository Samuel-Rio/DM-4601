import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#menunjukkan letak Data
dataset = pd.read_csv(r'C:\Users\acer\OneDrive\Desktop\Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values 

print(X)
print(Y)

#karena tidak ada data yang hilang/nan maka tidak perlu coding menghilangkan missing value

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [0])],remainder='passthrough')
x = np.array(ct.fit_transform(X))


#merubah matriks Y menjadi nilai numerik 1,2,3 dst dengan labelencoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

print("Ini hasil LabelEncoder",Y)

#membagi dataset menjadi 2 set (training set dan test set)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1)
print("Ini X train",X_train)
print("Ini X test",X_test)
print("Ini Y train",Y_train)
print("Ini Y test",Y_test)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print("Ini hasil scaling X train",X_train)
print("Ini hasil scaling X test",X_test)