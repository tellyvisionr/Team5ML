import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and Explore Data
symptom_data = pd.read_csv("Covid 19 symptoms data UPDATED.csv")
covid_data = pd.read_csv("Covid Data UPDATED.csv")

# Step 2: Merge datasets using common identifiers
merged_data = pd.merge(symptom_data, covid_data, on=['Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+', 'SEX', 'HAS_COVID'])

# Handle missing or inconsistent data (if any)

# Step 3: Feature Selection
X = merged_data[['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
                 'None_Symptom', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+',
                 'SEX', 'DIABETES', 'OBESITY', 'ASTHMA', 'TOBACCO']]

# Define the target variable
y = merged_data['HAS_COVID']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_prob = model.predict_proba(X_test)[:, 1]

# Default threshold evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Default Accuracy: {accuracy}")
print("Default Classification Report:")
print(report)

# Adjusting the threshold
threshold = 0.4  # Adjust the threshold as needed
y_pred_adjusted = (y_prob > threshold).astype(int)

# Evaluate the adjusted predictions
accuracy_adjusted = accuracy_score(y_test, y_pred_adjusted)
report_adjusted = classification_report(y_test, y_pred_adjusted)

print(f"Adjusted Accuracy: {accuracy_adjusted}")
print("Adjusted Classification Report:")
print(report_adjusted)

# Step 7: Make Predictions (on new data)
new_data = pd.DataFrame({
    'Fever': [1],
    'Tiredness': [0],
    'Dry-Cough': [1],
    'Difficulty-in-Breathing': [0],
    'Sore-Throat': [1],
    'None_Symptom': [0],
    'Age_0-9': [0],
    'Age_10-19': [1],
    'Age_20-24': [0],
    'Age_25-59': [0],
    'Age_60+': [0],
    'SEX': [1],  # Assuming female
    'DIABETES': [0],
    'OBESITY': [1],
    'ASTHMA': [0],
    'TOBACCO': [0]
})

# Make predictions
new_predictions = model.predict(new_data)

# Print predictions
print("Predictions for new data:")
print(new_predictions)
