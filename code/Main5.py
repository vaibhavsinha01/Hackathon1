import os
import pandas as pd

data = pd.read_csv(r'C:\Users\Vaibhav\Desktop\hackathon2\data\storage\BankChurners.csv')

old_customers_df = data[data['Attrition_Flag'] == 0]
old_customers_df.to_csv(os.path.join(r'C:\Users\Vaibhav\Desktop\hackathon2\data\report', 'Old_Customers.csv'))

items = data[["Gender", "Dependent_count", "Education_Level", "Marital_Status", "Income_Category", "Card_Category", 
              "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon"]]

for item in items.columns:
    mode_value = data[item].mode()[0]  
    print(f'Mode of {item}: {mode_value}')