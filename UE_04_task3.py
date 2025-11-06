"""
UE_04.py
----------------------------
Simplified diagnostic visualization for OLS regression models.
Author: Aditya Aiya
"""

import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_ols_diagnostics(model, save_path="UE_04_App2_DiagnosticPlots.pdf"):
    """
    Generate 4 standard OLS diagnostic plots:
    1. Residuals vs Fitted
    2. Normal Q-Q Plot
    3. Scale-Location Plot
    4. Leverage vs Residuals (Cook's Distance)
    """

    # Extract fitted values and residuals
    fitted_vals = model.fittedvalues
    residuals = model.resid

    # Create 2x2 plot grid
    fig = plt.figure(figsize=(10, 10))

    # === 1) Residuals vs Fitted ===
    plt.subplot(2, 2, 1)
    plt.scatter(fitted_vals, residuals, color='purple', alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title("Residuals vs Fitted")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")

    # === 2) Normal Q-Q Plot ===
    plt.subplot(2, 2, 2)
    sm.qqplot(residuals, line='s', ax=plt.gca())
    plt.title("Normal Q-Q")

    # === 3) Scale-Location ===
    plt.subplot(2, 2, 3)
    plt.scatter(fitted_vals, abs(residuals)**0.5, color='teal', alpha=0.6)
    plt.title("Scale-Location")
    plt.xlabel("Fitted values")
    plt.ylabel("√|Residuals|")

    # === 4) Leverage vs Residuals ===
    plt.subplot(2, 2, 4)
    sm.graphics.influence_plot(model, ax=plt.gca(), criterion="cooks")
    plt.title("Leverage vs Residuals")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Diagnostic plots saved as '{save_path}'")
