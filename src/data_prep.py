import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

# 1. Load Data (Assuming 'data.csv' exists in your /data folder)
df = pd.read_csv('data/creditcard.csv')

# 2. EDA: Check imbalance
print(df['Class'].value_counts(normalize=True)) 

# 3. Preprocessing: Scaling Time and Amount (RobustScaler handles outliers well)
scaler = RobustScaler()
df['Amount_Scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time_Scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# 4. Feature Selection & Train/Test Split
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Handle Imbalance (SMOTE applied ONLY to training data to avoid data leakage)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)