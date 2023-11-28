import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and Explore Data
symptom_data = pd.read_csv("Covid 19 symptoms data UPDATED.csv")
covid_data = pd.read_csv("Covid Data UPDATED.csv")

# Randomize the data
symptom_data_randomized = symptom_data.sample(frac=1, random_state=20)
covid_data_randomized = covid_data.sample(frac=1, random_state=20)

# Reduce the size to 100 rows each
symptom_data_sampled = symptom_data_randomized.head(100)
covid_data_sampled = covid_data_randomized.head(100)

# Merge datasets using common identifiers
merged_data = pd.merge(symptom_data_sampled, covid_data_sampled, on=['Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+', 'SEX'])

# Feature Selection
X = merged_data[['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
                 'None_Sympton', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+',
                 'Severity_None', 'Severity_Mild', 'Severity_Moderate', 'Severity_Severe',
                 'SEX', 'DIABETES', 'OBESITY', 'ASTHMA', 'TOBACCO']]
y = merged_data['HAS_COVID']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define an extended search space (can be adjusted based on your specific requirements)
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
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
print("Best Parameters:", random_search.best_params_)

# Build and Train the Model with the best parameters
best_model_random = random_search.best_estimator_
best_model_random.fit(X_train, y_train)

# Evaluate the Model
y_pred_random = best_model_random.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)
report_random = classification_report(y_test, y_pred_random)

print(f"Accuracy (RandomizedSearchCV): {accuracy_random}")
print("Classification Report:")
print(report_random)
