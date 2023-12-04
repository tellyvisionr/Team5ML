import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the first 200 rows of the merged data directly
merged_data = pd.read_csv("merged_data.csv")

# Shuffle the rows randomly
merged_data_randomized = merged_data.sample(frac=1, random_state=20)

# Select the first 100 rows
merged_data_subset = merged_data_randomized.head(100)

# Feature Selection
X = merged_data_subset[['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
                        'None_Symptom', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+',
                        'Severity_None', 'Severity_Mild', 'Severity_Moderate', 'Severity_Severe',
                        'SEX', 'DIABETES', 'OBESITY', 'ASTHMA', 'TOBACCO']]
y = merged_data_subset['HAS_COVID']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Define an extended search space (can be adjusted based on your specific requirements)
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # Adjust the subsample values
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use only the Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)

# Use RandomizedSearchCV with parallel processing
random_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5, n_jobs=-1)
random_search.fit(X_train, y_train)

# Print the best parameters found by RandomizedSearchCV
print(f"Model: {model.__class__.__name__}")
print("Best Parameters (RandomizedSearchCV):", random_search.best_params_)

# Build and Train the Model with the best parameters from RandomizedSearchCV
best_model_random = random_search.best_estimator_
best_model_random.fit(X_train, y_train)

# Evaluate the Model on the original test set
y_pred_random = best_model_random.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)
report_random = classification_report(y_test, y_pred_random)

print(f"Accuracy (RandomizedSearchCV): {accuracy_random}")
print("Classification Report:")
print(report_random)
