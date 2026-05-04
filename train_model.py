import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# 1. Load your approved dataset
df = pd.read_csv('WineQT.csv')

# 2. Data Cleaning: Drop 'Id' as it has no predictive value
# and separate features (X) from the target (y)
X = df.drop(['quality', 'Id'], axis=1)
y = df['quality']

# 3. Apply SMOTE to balance the dataset
# We use k_neighbors=1 to handle the very small number of samples in classes 3 and 8
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Original dataset shape:", y.value_counts().to_dict())
print("Resampled dataset shape:", y_resampled.value_counts().to_dict())

# 4. Train the Finalized Random Forest Model
# Random Forest was chosen for its high accuracy and stability
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# 5. Save the model as a .pkl file for your Flask UI
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nSUCCESS: Final model trained with SMOTE and saved as model.pkl")