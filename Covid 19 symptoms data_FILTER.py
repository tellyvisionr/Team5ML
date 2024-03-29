import pandas as pd

# Step 1: Load and Explore Data
symptoms_data = pd.read_csv("Covid 19 symptoms data.csv")  # Replace with the actual file path

# Step 2: Create 'SEX' column based on 'Gender_Female' and 'Gender_Male'
symptoms_data['SEX'] = symptoms_data['Gender_Female'].astype(int)

# Step 3: Drop the original 'Gender_Female' column
symptoms_data = symptoms_data.drop('Gender_Female', axis=1)

# Step 4: Reduce the number of rows to 1000
symptoms_data = symptoms_data.sample(n=1000, random_state=42)

# Step 5: Remove specified columns
remove_columns = ['Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea', 'None_Experiencing', 'Contact_Dont-Know', 'Country']
symptoms_data = symptoms_data.drop(remove_columns, axis=1)

# Print sample data to verify the changes
print(symptoms_data.head())

# Step 8: Save the modified dataset if needed
symptoms_data.to_csv("Covid 19 symptoms data UPDATED.csv", index=False)  # Replace with the desired file path
