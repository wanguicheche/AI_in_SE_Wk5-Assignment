# Predicting patient readmission within 30 days

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# --- Step 1: Create sample data ---
data = {
    'age': [45, 60, 30, 70, 50],
    'blood_pressure': [120, 150, 110, 160, 140],
    'days_in_hospital': [3, 7, 2, 10, 5],
    'readmitted': [0, 1, 0, 1, 1]  # 1 = readmitted, 0 = not readmitted
}
df = pd.DataFrame(data)
print("Sample Data:\n", df)

# --- Step 2: Split into features and target ---
X = df[['age', 'blood_pressure', 'days_in_hospital']]
y = df['readmitted']
print(df)


# --- Step 3: Split into training and test sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Step 4: Build and train the Logistic Regression model ---
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Step 5: Make predictions ---
y_pred = model.predict(X_test)

# --- Step 6: Evaluate the model ---
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)
