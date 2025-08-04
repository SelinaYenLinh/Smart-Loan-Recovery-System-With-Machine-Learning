from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

class riskclassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100,random_state= 42)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    def evaluation(self, x_test, y_test):
        preds = self.model.predict(x_test)
        report = classification_report(y_test, preds)
        return report

