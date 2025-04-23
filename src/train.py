# Train a simple model on the iris dataset
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
dump(model, 'models/model.pkl')
