import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv(r'C:\Users\Vaibhav\Desktop\hackathon1\BankChurners.csv')

# Preprocessing
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

# Features and target
X = data[["Customer_Age", "Gender", "Dependent_count", "Education_Level", "Marital_Status", "Income_Category", 
          "Card_Category", "Months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon", 
          "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy", 
          "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", 
          "Avg_Utilization_Ratio"]]

y = data["Attrition_Flag"]

# Create subplots
fig, axes = plt.subplots(4, 5, figsize=(20, 16))  
fig.suptitle('Feature Scatterplots and Histograms vs Attrition_Flag')

axes = axes.flatten()

# Features for plotting
features = ["Customer_Age", "Gender", "Dependent_count", "Education_Level", "Marital_Status", "Income_Category", 
            "Card_Category", "Months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon", 
            "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy", 
            "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", 
            "Avg_Utilization_Ratio"]

for i, feature in enumerate(features):
    if i < len(axes):  
        sns.scatterplot(x=data[feature], y=data["Attrition_Flag"], ax=axes[i], alpha=0.6)
        axes[i].set_title(f"{feature} vs Attrition_Flag")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Attrition_Flag")
        sns.histplot(data[feature], bins=20, ax=axes[i], color='blue', kde=True)

if len(features) < len(axes):
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.3, hspace=0.4)

plt.show()
