from nexgml.gradient_supported import BasicRegressor
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from nexgml.metrics import r2_score, accuracy_score
from nexgml.amo.forlinear import mean_absolute_error
import time

X, y = make_regression(n_samples=100000, n_features=200, n_informative=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

start = time.time()
model = BasicRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"TIME: {time.time() - start}")
print(f"MAE: {mean_absolute_error(y_test, pred)}")
print(f"R^2: {r2_score(y_test, pred)}")