import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load the CSV file
data = pd.read_csv("/tmp/dataset01.csv")

# Extract columns
x = data["x"]
y = data["y"]

# 1. Number of data entries
print("Number of entries in y:", len(y))

# 2. Mean
print("Mean of y:", np.mean(y))

# 3. Standard deviation
print("Standard deviation of y:", np.std(y, ddof=1))

# 4. Variance
print("Variance of y:", np.var(y, ddof=1))

# 5. Min and Max
print("Min of y:", np.min(y))
print("Max of y:", np.max(y))

# 6. OLS model (y = a + bx)
x_with_const = sm.add_constant(x)  # adds intercept
model = sm.OLS(y, x_with_const).fit()
print("\nOLS Summary:")
print(model.summary())

# Store OLS model
import pickle
with open("/tmp/OLS_model", "wb") as f:
    pickle.dump(model, f)
print("\nOLS model stored as 'OLS_model'")
