import numpy as np
a=[[1,2,3,4],[2,3,4,5],[6,7,9,0]]
print(a)
print(a[0])
# Convert 'a' to a NumPy array first, then index
print(np.array(a)[0,:])
print(a[0][3])
k=2
# Convert 'a' to a NumPy array, then slice the k-th column
res=np.array(a)[:,k]
print(res)
array=[18,21,39,48,53,60,73,85]
print(array)
import pandas as pd
d={'col':[1,2,3,4],'col1':[1,7,3,6],'col2':[9,2,8,3]}
df=pd.DataFrame(d)
print(df)
df.shape
nrows=df.index.size
print(nrows)
ncol=len(df.columns)
print(ncol)
print(df.shape[1])
print(df.shape[0])
print(df.head(3))
print(df.tail)
print(list(df.columns))
print(df.head(3).mean())
print(df.info())
print(df.describe())
print(df.mean())
print(df.median())
print(df.mode())
print(df.std())
print(df.var())
print(df.min())
print(df.max())
print(df.col2.mean())
print(df.col1.median())
print(df.info())
print(df.describe().T)
import pandas as pd
import numpy as np
d={'col':[1,2,np.nan,4],'col1':[1,np.nan,3,6],'col2':[9,2,8,np.nan]}
df2=pd.DataFrame(d)
print(df2.isnull())
print(df2.notnull())
print(df2.info())
print(df2.isnull().sum())
print(df2.isnull().sum().sum())
print(df.dtypes)


import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt
df=pd.read_csv("diabetes.csv")
print(df.head())
df.describe().T
df.info()
df.shape
df.Age.isnull().sum()

#histogram
df.hist(bins=20,figsize=(15,10),color="teal",edgecolor='b')
plt.suptitle("Histogram for each data")
plt.show()

#boxplot
fig, axs = plt.subplots(9, 1, figsize=(7, 17), dpi=95)
i = 0
for col in df.columns:
    axs[i].boxplot(df[col], vert=False)
    axs[i].set_ylabel(col)
    i += 1
plt.tight_layout() # Adjust layout to prevent overlapping
plt.show()

df.boxplot(column=col, vert=False)
plt.suptitle("Box plot for the 'Outcome' column")
plt.show()

#heatmap
corr=df.corr()
sns.heatmap(df.corr(),annot=True,fmt=".2f",cmap="pink")
plt.show()


plt.pie(df.Outcome.value_counts(),labels=["Not diabetes",'diabetes'],autopct='%.2f')
plt.show()

x=df.drop(columns=["Outcome"])
y=df.Outcome
x.head()

x=df.drop(columns=["Outcome"])
y=df.Outcome
print(x.head)
print(y.head)

#standaredscaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)
print(x)

#minmax scaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x)
print(x)


#GridSearchCV
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

iris=datasets.load_iris()
x=iris.data
y=iris.target

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=SVC()

hyperparam_grid={
    'C':[0.1,1,10,100],'kernel':['linear','rbf','poly'],
    'gamma':['scale','auto']}
grid_search=GridSearchCV(estimator=model,param_grid=hyperparam_grid,scoring='accuracy',cv=5,verbose=2,n_jobs=-1)

grid_search.fit(X_train,Y_train)
best_hyperparam=grid_search.best_params_
best_score=grid_search.best_score_

print("Best Hyperparameters:",best_hyperparam)
print("Best Cross-Validation Score:",best_score)
print(grid_search.best_params_)

best_model=grid_search.best_estimator_
y_pred=best_model.predict(X_test)
print("Accuracy:",accuracy_score(Y_test,y_pred))
print("Classifcation report:")
print(classification_report(Y_test,y_pred))

#randomizeedsearchcv
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

iris=datasets.load_iris()
x=iris.data
y=iris.target

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=SVC()

hyperparam_grid={
    'C':[0.1,1,10,100],'kernel':['linear','rbf','poly'],
    'gamma':['scale','auto']}
random_search=RandomizedSearchCV(estimator=model,param_distributions=hyperparam_grid,scoring='accuracy',cv=5,verbose=2,n_jobs=-1)

random_search.fit(X_train,Y_train)
best_hyperparam=random_search.best_params_
best_score=random_search.best_score_

print("Best Hyperparameters:",best_hyperparam)
print("Best Cross-Validation Score:",best_score)
print(random_search.best_params_)

best_model=random_search.best_estimator_
y_pred=best_model.predict(X_test)
print("Accuracy:",accuracy_score(Y_test,y_pred))
print("Classifcation report:")
print(classification_report(Y_test,y_pred))
