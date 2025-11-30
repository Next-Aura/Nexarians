from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from nexgml.gradient_supported import (BasicClassifier,
                                       BasicRegressor,
                                       IntenseRegressor,
                                       IntenseClassifier,
                                       L1Classifier,
                                       L1Regressor,
                                       L2Classifier,
                                       L2Regressor,
                                       ElasticNetClassifier,
                                       ElasticNetRegressor,
                                       SRClassifier,
                                       MRClassifier)
from nexgml.tree_models import (TreeBackendClassifier,
                                TreeBackendRegressor,
                                ForestBackendClassifier,
                                ForestBackendRegressor)
from nexgml.metrics import f1_score

X_c, y_c = make_classification(n_samples=10000, n_features=200, n_classes=3, n_informative=100)

X_r, y_r = make_regression(n_samples=10000, n_features=200, n_informative=100)

X_trainc, X_testc, y_trainc, y_testc = train_test_split(X_c, y_c, test_size=0.2)
X_trainr, X_testr, y_trainr, y_testr = train_test_split(X_r, y_r, test_size=0.2)

model = TreeBackendClassifier()
model.fit(X_trainc, y_trainc)
pred = model.predict(X_testc)
print("Score:", f1_score(y_testc, pred))