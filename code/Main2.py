import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
import joblib
import os

class Model:
    def __init__(self, model_path='strategies.joblib'):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
    
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
    
    def correlation_report(self):
        X = self.data[["Customer_Age", "Gender", "Dependent_count", "Education_Level", "Marital_Status", "Income_Category", "Card_Category", 
                    "Months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", 
                    "Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", 
                    "Avg_Utilization_Ratio"]]
        y = self.data['Attrition_Flag']

        correlation_values = X.apply(lambda col: col.corr(y))
        
        print("Correlation report:")
        print(correlation_values)
        val_df = pd.DataFrame(correlation_values)
        val_df.to_csv(os.path.join(r'data\report','Model_Correlation_Report.csv'))
    
    def plotting(self):
        X = self.data[["Customer_Age", "Gender", "Dependent_count", "Education_Level", "Marital_Status", "Income_Category", "Card_Category", 
                    "Months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", 
                    "Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", 
                    "Avg_Utilization_Ratio"]]
        y = self.data['Attrition_Flag']

        

    def preprocess_data(self):
        X = self.data[["Customer_Age", "Gender", "Dependent_count", "Education_Level", "Marital_Status", "Income_Category", "Card_Category", 
                       "Months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", 
                       "Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", 
                       "Avg_Utilization_Ratio"]]
        y = self.data["Attrition_Flag"]

        X_scaled = self.scaler.fit_transform(X)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=3)

    def train_model(self):
        self.model = XGBClassifier()
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
        self.preprocess_data()
        self.train_model()
        self.evaluate_metrics()
        self.dump_model()

if __name__ == "__main__":
    churn_model = Model()  
    churn_model.run(r'C:\Users\Vaibhav\Desktop\hackathon2\data\storage\BankChurners.csv')  

    # To access the trained model:
    # churn_model.model  # This gives access to the trained model within the instance
