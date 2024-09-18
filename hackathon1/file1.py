import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report, confusion_matrix,f1_score
import joblib

data = pd.read_csv(r'C:\Users\Vaibhav\Desktop\hackathon1\BankChurners.csv')

data['Attrition_Flag'] = np.where(data['Attrition_Flag'] == 'Existing Customer', 1, 0)
data['Gender'] = np.where(data['Gender'] == 'M', 1, 0)

education_mapping = {'Uneducated': 0, 'High School': 1, 'College': 2, 'Unknown': 3, 'Graduate': 4, 'Post-Graduate': 5, 'Doctorate': 6}
data['Education_Level'] = data['Education_Level'].map(education_mapping)

marital_status_mapping = {'Single': 0, 'Divorced': 1, 'Unknown': 2, 'Married': 3}
data['Marital_Status'] = data['Marital_Status'].map(marital_status_mapping)

income_category_mapping = {'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2, 'Unknown': 3, '$80K - $120K': 4, '$120K +': 5}
data['Income_Category'] = data['Income_Category'].map(income_category_mapping)

card_category_mapping = {'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
data['Card_Category'] = data['Card_Category'].map(card_category_mapping)

print(data.head())

# X = data[["Customer_Age", "Gender", "Dependent_count", "Education_Level", "Marital_Status", "Income_Category", "Card_Category", "Months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"]]
X = data[["Customer_Age", "Gender", "Dependent_count", "Education_Level", "Marital_Status", "Income_Category", "Card_Category", "Months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"]]
y = data["Attrition_Flag"]

scaler = StandardScaler()
# scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=3)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3) # maybe there is some error with the scaler function check later

# model = RandomForestClassifier()
# model = SVC()
model = XGBClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test,y_pred))
print("Recall", recall_score(y_pred,y_test))
print('F1score', f1_score(y_pred,y_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model,'strategies.joblib')

