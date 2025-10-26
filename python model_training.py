# ===============================
# 📦 REQUIRED LIBRARIES
# ===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ LOAD DATA
# ===============================
df = pd.read_csv('manufacturing_dataset_1000_samples.csv', parse_dates=['Timestamp'])

# Force datetime conversion (in case pandas missed it)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# ===============================
# 2️⃣ QUICK CHECKS
# ===============================
print("✅ Data Loaded Successfully!")
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

# ===============================
# 3️⃣ FEATURE ENGINEERING
# ===============================
if 'Timestamp' in df.columns:
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df = df.drop(columns=['Timestamp'])

# ===============================
# 4️⃣ DEFINE TARGET & FEATURES
# ===============================
target = 'Parts_Per_Hour'
X = df.drop(columns=[target])
y = df[target]

print(f"\n📊 Average Parts_Per_Hour: {y.mean():.2f}")

# ===============================
# 5️⃣ TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ===============================
# 6️⃣ PREPROCESSING PIPELINE
# ===============================
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ===============================
# 7️⃣ MODEL PIPELINE
# ===============================
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# ===============================
# 8️⃣ TRAIN MODEL
# ===============================
model.fit(X_train, y_train)
print("\n✅ Model training complete!")

# ===============================
# 9️⃣ PREDICT & EVALUATE
# ===============================
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 Model Performance:")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R² Score: {r2:.3f}")

# ===============================
# 🔟 RESIDUAL PLOT
# ===============================
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.show()

# ===============================
# 1️⃣1️⃣ SAVE MODEL
# ===============================
joblib.dump(model, 'linear_regression_model.pkl')
print("\n💾 Model saved as 'linear_regression_model.pkl'")

# ===============================
# 1️⃣2️⃣ LOAD MODEL (Demonstration)
# ===============================
loaded_model = joblib.load('linear_regression_model.pkl')

# ===============================
# 1️⃣3️⃣ PREDICT ON NEW DATA
# ===============================
# Create a sample row based on average of numeric columns and most frequent categorical values
new_data = pd.DataFrame({
    col: [X[col].mode()[0]] if X[col].dtype == 'O' else [X[col].mean()]
    for col in X.columns
})

print("\n🧩 New data sample used for prediction:")
print(new_data)

# Make prediction
predicted_value = loaded_model.predict(new_data)[0]

print(f"\n🧠 Predicted Parts_Per_Hour for new data: {predicted_value:.2f}")
print(f"📊 Dataset average Parts_Per_Hour: {y.mean():.2f}")
print(f"🔍 Difference: {abs(predicted_value - y.mean()):.2f}")