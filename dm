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



import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

np.random.seed(42)

x=np.linspace(0,10,100)
y=np.sin(x)+np.random.normal(0,0.3,size=x.shape)

X_reshaped =x.reshape(-1,1)

model_overfit=make_pipeline(PolynomialFeatures(degree=15),LinearRegression())
model_overfit.fit(X_reshaped,y)

y_pred_overfit=model_overfit.predict(X_reshaped)

plt.figure(figsize=(8,6))
plt.scatter(x,y,color='blue',label='Data')
plt.plot(x,y_pred_overfit,color='orange',label='Overfitting (Polynomial Degree)')
plt.title("Overfitting Example")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X_reshaped,y)

y_pred=model.predict(X_reshaped)
df=pd.DataFrame(y_pred)
print(df.head(10))

plt.figure(figsize=(8,6))
plt.scatter(x,y,color='blue',label='Data')
plt.plot(x,y_pred,color='orange',label='Underfitting (Linear Regression)')
plt.title("Underfitting Example")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

print('classification report')
print(classification_report(actual,predicted))



# Binary classification
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

actual=np.array(['Dog','Dog','Dog','Not Dog','Dog','Not Dog','Dog','Dog','Not Dog','Not Dog'])
predicted=np.array(['Dog','Not Dog','Dog','Not Dog','Dog','Dog','Dog','Dog','Not Dog','Not Dog'])

cm=confusion_matrix(actual,predicted)

sns.heatmap(cm,annot=True,xticklabels=['Dog','Not Dog'],yticklabels=['Dog','Not Dog'],cmap='Blues',fmt='g')
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Predicted',fontsize=13)
plt.gca().xaxis.tick_top()
plt.ylabel('Actual',fontsize=13)
plt.title("Confusion Matrix",fontsize=15,pad=20)
plt.show()
print('classification report')
print(classification_report(actual,predicted))





#Multiclass classification
import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt

y_true=['Cat']*10+['Dog']*12+['Horse']*10
print(y_true)

y_pred=['Cat']*8+['Dog']+['Horse']+['Cat']*2+['Dog']*10+['Horse']*8+['Dog']*2
print(y_pred)
classes=['Cat','Dog','Horse']

cm=confusion_matrix(y_true,y_pred,labels=classes)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)
disp.plot(cmap=plt.cm.Greens)
plt.title('Confusion Matrix',fontsize=13,pad=20)
plt.xlabel('Predicted',fontsize=11)
plt.ylabel('Actual',fontsize=11)

plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.show()

print('classification report')
print(classification_report(y_true,y_pred))

# 0.80+0.77+0.89/3 -> macro avg
# 0.80*10 + 0.77*12 + 0.89*10 -> weighted avg



import matplotlib.pyplot as plt
from scipy import stats
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
slope,intercept,r,p,std_err=stats.linregress(x,y)
# yhat - myfunc
def myfunc(x):
  return slope*x+intercept
mymodel=list(map(myfunc,x))
print(mymodel)
print("Correlation coefficient:",r)
yhat = myfunc(10)
print("Predicted value at x=10 is",yhat)
plt.scatter(x,y)
plt.scatter(10,yhat)
plt.plot(x,mymodel)
plt.xlabel('x',fontsize=13)
plt.ylabel('y',fontsize=13)
plt.title('Linear Regression',fontsize=14,pad=20)
plt.show()




import numpy as np
import matplotlib.pyplot as plt
x= [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel=np.poly1d(np.polyfit(x,y,3))
yhat=mymodel(10.5)
print('Predicted value at 10.5 is',yhat)
plt.scatter(x,y)
plt.scatter(10.5,yhat)
plt.plot(x,mymodel(x))
plt.xlabel('x',fontsize=13)
plt.ylabel('y',fontsize=13)
plt.title('Polynomial Regression',fontsize=14,pad=20)
plt.show()




#EX 9 AND 10
# KMEANS AND AGGLOMERATIVE CLUSTERING
# -------------------------------
# K-MEANS CLUSTERING

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Data
x = [5, 6, 12, 5, 4, 13, 15, 7, 10, 14]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 27]

data = list(zip(x, y))

print("Data points:")
print(data)

# Apply KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print("\nCentroids:")
print(centroids)

print("\nLabels:")
print(labels)

# Plot
plt.scatter(x, y, c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='X', color='red')

plt.title("K-Means Clustering")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()


# -------------------------------
# AGGLOMERATIVE CLUSTERING

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Data
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 14, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

print("\nData points:")
print(data)

# Dendrogram
linkage_data = linkage(data, method='complete', metric='euclidean')

dendrogram(linkage_data)

plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Agglomerative clustering
hierarchical_cluster = AgglomerativeClustering(
    n_clusters=2, linkage='complete'
)

labels = hierarchical_cluster.fit_predict(data)

print("\nCluster Labels:")
print(labels)

# Plot
plt.scatter(x, y, c=labels)

plt.title("Agglomerative Clustering (Complete Linkage)")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()
