import pandas as pd

# Step 1: Load and Explore Data
merged_data = pd.read_csv("Covid Data.csv")  # Replace with the actual file path

# Step 2: Replace 'CLASSIFICATION_FINAL' with 'HAS_COVID'
merged_data['HAS_COVID'] = (merged_data['CLASIFFICATION_FINAL'] <= 3).astype(int)

# Step 3: Drop the original 'CLASSIFICATION_FINAL' column if needed
merged_data = merged_data.drop('CLASIFFICATION_FINAL', axis=1)

# Step 4: Add Age-related Columns
merged_data['Age_0-9'] = (merged_data['AGE'].between(0, 9)).astype(int)
merged_data['Age_10-19'] = (merged_data['AGE'].between(10, 19)).astype(int)
merged_data['Age_20-24'] = (merged_data['AGE'].between(20, 24)).astype(int)
merged_data['Age_25-59'] = (merged_data['AGE'].between(25, 59)).astype(int)
merged_data['Age_60+'] = (merged_data['AGE'] >= 60).astype(int)

# Step 5: Drop the original 'AGE' column
merged_data = merged_data.drop('AGE', axis=1)

# Step 6: Reduce the number of rows to 1000
merged_data = merged_data.sample(n=1000, random_state=42)

# Step 7: Remove specified columns
remove_columns = ['USMER', 'MEDICAL_UNIT', 'DATE_DIED', 'INTUBED', 'PNEUMONIA', 'PREGNANT', 
                  'COPD', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 
                  'RENAL_CHRONIC', 'ICU']
merged_data = merged_data.drop(remove_columns, axis=1)

# Print sample data to verify the changes
print(merged_data.head())

# Step 8: Save the modified dataset if needed
merged_data.to_csv("Covid Data UPDATED.csv", index=False)  # Replace with the desired file path
