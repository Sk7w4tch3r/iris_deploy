import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



iris_dataset = load_iris()

X_train, X_test, y_train , y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)


print("train accuracy is ", knn.score(X_train, y_train))
print("test accuracy is ", knn.score(X_test, y_test))


target_names = ['setosa', 'versicolor', 'virginica']

joblib.dump({"model": knn, "target_names": target_names}, 'knn.joblib')

print("model saved to knn.joblib")