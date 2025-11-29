from sklearn.datasets import make_classification
from nexgml.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from nexgml.gradient_supported import BasicClassifier

X, y = make_classification(n_samples=10000, n_features=100, n_classes=3, n_informative=50, scale=300000.0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = BasicClassifier(learning_rate=0.05, verbose=2, verbosity='heavy', lr_scheduler='adaptive')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))