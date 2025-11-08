import pandas as pd
from supabase_client import SupabaseDatabase

# Load dataset (change path if needed)
df = pd.read_csv("heart.csv")

# Optional — check columns
print("\n--- Columns in CSV ---")
print(df.columns.tolist())

# Ensure correct column names
required_cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal','target']

if not all(col in df.columns for col in required_cols):
    raise ValueError("CSV does not have all required columns!")

# Convert DataFrame to list of dicts for Supabase
data_list = df.to_dict(orient='records')

# Upload
db = SupabaseDatabase()
response = db.insert_heart_disease_data(data_list)

print("\n✅ Data uploaded successfully!")
print(f"Total rows inserted: {len(response)}")
