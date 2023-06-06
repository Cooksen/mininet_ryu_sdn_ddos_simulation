from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRFClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings("ignore")

class MachineLearning():

    def __init__(self):
        print("data loading...")
        self.flow_dataset = pd.read_csv('./FlowRecordFile.csv')

        self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace('.', '')
        self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace('.', '')       

    def feature_encoding(self):
        categorical_columns = ['ip_src', 'ip_dst', 'flags']

        label_encoders = {}
        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            self.flow_dataset[column] = label_encoders[column].fit_transform(self.flow_dataset[column])

    def flow_training(self):
        print("training...")
        
        X_flow = self.flow_dataset.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')

        y_flow = self.flow_dataset.iloc[:, -1].values

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        classifier = XGBRFClassifier(n_estimators=10, max_depth=6, learning_rate=0.1)
        flow_model = classifier.fit(X_flow_train, y_flow_train)

        y_flow_pred = flow_model.predict(X_flow_test)

        print("------------------------------------------------------------------------------")

        print("confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        print(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)

        print("success accuracy = {0:.2f} %".format(acc * 100))
        fail = 1.0 - acc
        print("fail accuracy = {0:.2f} %".format(fail * 100))
        print("------------------------------------------------------------------------------")

        x = ['TP', 'FP', 'FN', 'TN']
        plt.title("XGBRFClassifier")
        plt.xlabel('Predicted Class')
        plt.ylabel('Number of Flows')
        plt.tight_layout()
        y = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        plt.bar(x, y, color="#000000", label='XGBRFClassifier')
        plt.legend()
        # plt.show()

def main():
    start = datetime.now()
    
    ml = MachineLearning()
    ml.feature_encoding()
    ml.flow_training()

    end = datetime.now()
    print("Training time: ", (end - start))

if __name__ == "__main__":
    main()
