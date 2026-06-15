import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

data = {
    'study_hours_per_day': np.round(np.random.uniform(0, 10, n), 1),
    'attendance_percentage': np.round(np.random.uniform(40, 100, n), 1),
    'previous_score': np.round(np.random.uniform(20, 100, n), 1),
    'sleep_hours': np.round(np.random.uniform(4, 10, n), 1),
    'extra_activities': np.random.randint(0, 2, n),
    'internet_access': np.random.randint(0, 2, n),
    'family_support': np.random.choice(['low', 'medium', 'high'], n),
    'gender': np.random.choice(['Male', 'Female'], n),
}

df = pd.DataFrame(data)

# Generate marks based on features
df['marks'] = (
    df['study_hours_per_day'] * 3.5 +
    df['attendance_percentage'] * 0.3 +
    df['previous_score'] * 0.25 +
    df['sleep_hours'] * 1.2 +
    df['extra_activities'] * 2 +
    df['internet_access'] * 1.5 +
    df['family_support'].map({'low': 0, 'medium': 3, 'high': 6}) +
    np.random.normal(0, 5, n)
).clip(0, 100).round(2)

df['pass_fail'] = (df['marks'] >= 40).astype(int)

df.to_csv('data/student_data.csv', index=False)
print("Dataset generated: data/student_data.csv")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Pass rate: {df['pass_fail'].mean()*100:.1f}%")