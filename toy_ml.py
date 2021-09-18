from sklearn import datasets

data = datasets.load_breast_cancer()

print(data.keys())

# Import pandas
import pandas as pd

# Read the DataFrame, first using the feature data
df = pd.DataFrame(data.data, columns=data.feature_names)

#let us keep the first five features
df = df[['mean radius' , 'mean texture' , 'mean perimeter'  , 'mean area'  , 'mean smoothness']]

# Add a target column, and fill it with the target data
df['target'] = data.target

# Show the first five rows
df.head()

df.info()

# Store the feature data
X = df.drop(columns = ['target'])   #data.data

# store the target data
y = df.target

# split the data using Scikit-Learn's train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.neighbors import KNeighborsClassifier
logreg = KNeighborsClassifier(n_neighbors=6)
logreg.fit(X_train, y_train)
logreg.score(X_test, y_test)

import pickle
#Save the Model to file in the current working directory

Pkl_Filename = "model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(logreg, file)