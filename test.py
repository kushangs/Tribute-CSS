# Practical 9 & 10 linearly separated & non linearly separated

from sklearn.datasets import load_iris
iris=load_iris()
print(iris)

print(iris.DESCR)

X=iris.data
y=iris.target
features = iris.feature_names
import seaborn as sns
import matplotlib.pyplot as plt
for i in range(4):
    sns.boxplot(x=y, y=X[:,i])
    plt.ylabel(features[i])
    plt.show()


for i in range(4):
  plt.hist(X[:,i],edgecolor='black')
  plt.title(features[i])
  plt.show()

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=32)

kernel=['linear','rbf','poly','sigmoid']
for i in kernel:
    model=SVC(kernel=i, C=1.0)
    model.fit(X_train, y_train
    print('For kernel:',i)
    print('Accuracy is :', model.score(X_test,y_test))

model=SVC()
model.fit(X_train, y_train)
print('accuracy on test data is:',model.score(X_test, y_test))
print('accuracy on train data is:', model.score(X_train, y_train))


for i in range(1,10):
    model=SVC(kernel='poly',degree=1, C=100)
    model.fit(X_train, y_train)
    print('accuracy on test data is:',model.score(X_test, y_test))
    print('accuracy on train data is:', model.score(X_train, y_train))

from sklearn.model_selection import GridSearchCV

param_grid={'C':[0.1,1,100,1000],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4,5,6]}
grid=GridSearchCV(SVC(),param_grid)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.score(X_test, y_test))
