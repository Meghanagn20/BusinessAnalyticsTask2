"""
UE_04 Application - Data Cleaning, Splitting, Visualization, and OLS Model
Author: Aditya Aiya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats
from UE_04 import plot_ols_diagnostics 
  # 👈 updated import

# ============================================================
# 1) Load Dataset
# ============================================================
df = pd.read_csv("dataset02.csv")

# Convert everything possible to numeric (ignore text)
df = df.apply(pd.to_numeric, errors='coerce').dropna()
print("✅ Columns in dataset:", list(df.columns))
print("Rows before cleaning:", len(df))

# ============================================================
# 2) Remove Outliers Safely using IQR filter
# ============================================================
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Keep only rows within 1.5 × IQR range
mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
df = df[mask].copy()
print("Rows after outlier removal:", len(df))

# ============================================================
# 3) Normalize both x and y
# ============================================================
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# ============================================================
# 4) Split Dataset (80% training / 20% testing)
# ============================================================
train_df, test_df = train_test_split(df_scaled, test_size=0.2, random_state=42)
train_df.to_csv("dataset02_training.csv", index=False)
test_df.to_csv("dataset02_testing.csv", index=False)

print("Training samples:", len(train_df), "| Testing samples:", len(test_df))

# ============================================================
# 5) OLS Model (without intercept)
# ============================================================
X_train = train_df[["x"]]   # independent variable
y_train = train_df["y"]     # dependent variable

ols_model = sm.OLS(y_train, X_train).fit()
print(ols_model.summary())

slope = ols_model.params.iloc[0]
print(f"Model Equation: y = {slope:.4f} × x")

# ============================================================
# 6) Scatter Plot (training vs testing + regression line)
# ============================================================
plt.figure(figsize=(8,6))
plt.scatter(train_df["x"], train_df["y"], color="orange", label="Training Data")
plt.scatter(test_df["x"], test_df["y"], color="blue", label="Testing Data")

x_line = np.linspace(df_scaled["x"].min(), df_scaled["x"].max(), 100)
y_line = slope * x_line
plt.plot(x_line, y_line, color="red", linewidth=2, label="OLS Fit")

plt.xlabel("x (normalized)")
plt.ylabel("y (normalized)")
plt.title("Scatter Plot with OLS Regression Line (No Intercept)")
plt.legend()
plt.tight_layout()
plt.savefig("UE_04_App2_ScatterVisualizationAndOlsModel.pdf")
plt.close()

# ============================================================
# 7) Boxplot of Normalized Data
# ============================================================
plt.figure(figsize=(6,5))
df_scaled.boxplot()
plt.title("Box Plot of Normalized Data")
plt.tight_layout()
plt.savefig("UE_04_App2_BoxPlot.pdf")
plt.close()

# ============================================================
# 8) Diagnostic Plots
# ============================================================
plot_ols_diagnostics(ols_model, save_path="UE_04_App2_DiagnosticPlots.pdf")

print("✅ All steps completed successfully.")
