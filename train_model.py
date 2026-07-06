import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names.tolist()  # ['setosa', 'versicolor', 'virginica']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Test accuracy: {accuracy:.3f}")

# Save both the model AND the class names together - the API needs the names
# to turn a predicted class index (0, 1, 2) back into a readable label
joblib.dump({"model": model, "target_names": target_names}, "iris_model.joblib")
print("Saved model to iris_model.joblib")
