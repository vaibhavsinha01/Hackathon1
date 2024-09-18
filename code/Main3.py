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
        self.data = None
    
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
        X1 = self.data[["Customer_Age", "Gender", "Dependent_count", "Education_Level"]]
        X2 = self.data[["Marital_Status", "Income_Category", "Card_Category", "Months_on_book"]]
        X3 = self.data[["Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit"]]
        X4 = self.data[["Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt"]]
        X5 = self.data[["Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"]]

        y = self.data['Attrition_Flag']

        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(12, 10))
        fig3, ((ax9, ax10), (ax11, ax12)) = plt.subplots(2, 2, figsize=(12, 10))
        fig4, ((ax13, ax14), (ax15, ax16)) = plt.subplots(2, 2, figsize=(12, 10))
        fig5, ((ax17, ax18), (ax19, ax20)) = plt.subplots(2, 2, figsize=(12, 10))

        ax1.hist(X1["Customer_Age"])
        ax1.set_title('Customer_Age')

        ax2.hist(X1["Gender"])
        ax2.set_title('Gender')

        ax3.hist(X1["Dependent_count"])
        ax3.set_title('Dependent_count')

        ax4.hist(X1["Education_Level"])
        ax4.set_title('Education_Level')

        ax5.hist(X2["Marital_Status"])
        ax5.set_title('Marital_Status')

        ax6.hist(X2["Income_Category"])
        ax6.set_title('Income_Category')

        ax7.hist(X2["Card_Category"])
        ax7.set_title('Card_Category')

        ax8.hist(X2["Months_on_book"])
        ax8.set_title('Months_on_book')

        ax9.hist(X3["Total_Relationship_Count"])
        ax9.set_title('Total_Relationship_Count')

        ax10.hist(X3["Months_Inactive_12_mon"])
        ax10.set_title('Months_Inactive_12_mon')

        ax11.hist(X3["Contacts_Count_12_mon"])
        ax11.set_title('Contacts_Count_12_mon')

        ax12.hist(X3["Credit_Limit"])
        ax12.set_title('Credit_Limit')

        ax13.hist(X4["Total_Revolving_Bal"])
        ax13.set_title('Total_Revolving_Bal')

        ax14.hist(X4["Avg_Open_To_Buy"])
        ax14.set_title('Avg_Open_To_Buy')

        ax15.hist(X4["Total_Amt_Chng_Q4_Q1"])
        ax15.set_title('Total_Amt_Chng_Q4_Q1')

        ax16.hist(X4["Total_Trans_Amt"])
        ax16.set_title('Total_Trans_Amt')

        ax17.hist(X5["Total_Trans_Ct"])
        ax17.set_title('Total_Trans_Ct')

        ax18.hist(X5["Total_Ct_Chng_Q4_Q1"])
        ax18.set_title('Total_Ct_Chng_Q4_Q1')

        ax19.hist(X5["Avg_Utilization_Ratio"])
        ax19.set_title('Avg_Utilization_Ratio')

        fig1.savefig(r'C:\Users\Vaibhav\Desktop\hackathon2\data\plotting\histograms_X1.png')
        fig2.savefig(r'C:\Users\Vaibhav\Desktop\hackathon2\data\plotting\histograms_X2.png')
        fig3.savefig(r'C:\Users\Vaibhav\Desktop\hackathon2\data\plotting\histograms_X3.png')
        fig4.savefig(r'C:\Users\Vaibhav\Desktop\hackathon2\data\plotting\histograms_X4.png')
        fig5.savefig(r'C:\Users\Vaibhav\Desktop\hackathon2\data\plotting\histograms_X5.png')

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
        self.plotting()
        self.preprocess_data()
        self.train_model()
        self.evaluate_metrics()
        self.dump_model()

if __name__ == "__main__":
    churn_model = Model()  
    churn_model.run(r'C:\Users\Vaibhav\Desktop\hackathon2\data\storage\BankChurners.csv') 


    # model_path = 'C:\\Users\\Vaibhav\\Desktop\\hackathon2\\strategies.joblib'
    # loaded_model = joblib.load(model_path)
    # predictions = loaded_model.predict(preprocessed_data) # put the preprocessed data here because normal data won't work
    # print("Predictions:", predictions)