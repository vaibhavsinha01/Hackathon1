# to analyze customer credit card data to forecast churn and develop strategies.

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import os

class Model:
    def __init__(self, model_path='strategies.joblib'):
        self.model = None
        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler()
        self.model_path = model_path
        self.data = None
        self.X = None
        self.y = None
    
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
    
    def set_attributes(self):
        self.data['Attrition_Flag'] = np.where(self.data['Attrition_Flag'] == 'Existing Customer', 1, 0)
        self.data['Gender'] = np.where(self.data['Gender'] == 'M', 1, 0)

        education_mapping = {'Uneducated': 0, 'High School': 1, 'College': 2, 'Unknown': 3, 'Graduate': 4, 'Post-Graduate': 5, 'Doctorate': 6}
        self.data['Education_Level'] = self.data['Education_Level'].map(education_mapping)

        marital_status_mapping = {'Single': 0, 'Divorced': 1, 'Unknown': 2, 'Married': 3}
        self.data['Marital_Status'] = self.data['Marital_Status'].map(marital_status_mapping)

        income_category_mapping = {'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2, 'Unknown': 3, '$80K - $120K': 4, '$120K +': 5}
        self.data['Income_Category'] = self.data['Income_Category'].map(income_category_mapping)

        card_category_mapping = {'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
        self.data['Card_Category'] = self.data['Card_Category'].map(card_category_mapping)
        self.data.to_csv(os.path.join(r'data\storage','newBankChurners.csv'))
    
    def correlation_report(self):
        # here we will not use the last column because there is a -0.99989 correlation to the y
        self.X = self.data[["Customer_Age", "Gender", "Dependent_count", "Education_Level", "Marital_Status", "Income_Category", "Card_Category", 
                            "Months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", 
                            "Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", 
                            "Avg_Utilization_Ratio"]]

        self.y = self.data['Attrition_Flag']

        correlation_values = self.X.apply(lambda col: col.corr(self.y))
        
        print("Correlation report:")
        print(correlation_values)
        val_df = pd.DataFrame(correlation_values)
        val_df.to_csv(os.path.join(r'data\report','Model_Correlation_Report.csv'))
    
    def plotting(self):
        features = ["Customer_Age", "Gender", "Dependent_count", "Education_Level",
                    "Marital_Status", "Income_Category", "Card_Category", "Months_on_book",
                    "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon",
                    "Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1",
                    "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"]

        plot_path = r'C:\Users\Vaibhav\Desktop\hackathon2\data\plotting'

        for feature in features:
            plt.figure(figsize=(10, 6))
            self.data[self.data['Attrition_Flag'] == 0][feature].hist(alpha=0.5, label='Existing Customer', color='blue')
            self.data[self.data['Attrition_Flag'] == 1][feature].hist(alpha=0.5, label='Attrited Customer', color='red')
            plt.title(f'{feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(os.path.join(plot_path, f'histogram_{feature}.png'))
            plt.close()

    def preprocess_data(self):
        X_scaled = self.scaler.fit_transform(self.X)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X_scaled, self.y, test_size=0.2, random_state=3)

    def train_model(self):
        # naive bayes not working well and xgb is the best performing among the 4 .
        self.model = XGBClassifier()
        # self.model = RandomForestClassifier()
        # self.model = SVC()
        self.model.fit(self.x_train, self.y_train)

    def evaluate_metrics(self):
        y_pred = self.model.predict(self.x_test)

        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Precision:", precision_score(self.y_test, y_pred))
        print("Recall:", recall_score(self.y_test, y_pred))
        print('F1score:', f1_score(self.y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def dump_model(self):
        joblib.dump(self.model, self.model_path)
        print(f"Model is saved at: {self.model_path}")
    
    def run(self, file_path):
        self.load_data(file_path)
        self.set_attributes()
        self.correlation_report()
        self.plotting()
        self.preprocess_data()
        self.train_model()
        self.evaluate_metrics()
        self.dump_model()

if __name__ == "__main__":
    churn_model = Model()  
    churn_model.run(r'C:\Users\Vaibhav\Desktop\hackathon2\data\storage\BankChurners.csv') 

# completed