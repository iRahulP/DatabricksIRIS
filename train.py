# Databricks notebook source
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

data = load_iris()

X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=10)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
plt.figure(2, figsize=(8,6))
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c = Y, cmap = plt.cm.Set1, edgecolor = 'k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

fig = plt.figure(1, figsize=(8,6))
ax = Axes3D(fig ,elev = -150, azim = 110)
X_reduced = PCA(n_components = 3).fit_transform(data.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c= Y, cmap = plt.cm.Set1, edgecolor = 'k', s = 40)
ax.set_title("3 PCA Directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

fig.savefig('iris1.png')

if __name__ == "__main__":
  with mlflow.start_run():
    dtc = DecisionTreeClassifier(random_state = 10)
    dtc.fit(X_train, Y_train)
    y_pred_class = dtc.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, y_pred_class)
    print(accuracy)
    
    mlflow.log_param("random_state", 10)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(dtc, "model")
    modelpath = "/dbfs/mlflow/iris/model-%s-%f" % ("decision_tree", 1)
    mlflow.sklearn.save_model(dtc, modelpath)
    mlflow.log_artifact("iris1.png")
