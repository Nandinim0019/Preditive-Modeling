🍲 Recipe Ingredient Prediction using Predictive Modeling
📌 Project Overview

This project focuses on predictive modeling for recipe ingredient scaling.
Given input recipes with ingredient quantities for different serving sizes, the model predicts ingredient requirements for new serving sizes using:

-Linear Scaling
-Proportional Scaling
-Power-Law Scaling

The predictions are then evaluated using standard error metrics (MAE, RMSE, MAPE, R²).

⚙️ Features
Parse and clean ingredient quantities from mixed formats (e.g., 2 nos., 1¾ nos. / 140 grams).
Predict ingredient quantities for unseen serving sizes.
Compare performance of three scaling approaches.
Evaluate models using error metrics.

🗂️ Project Structure
├── paneer_recipes.json        # Sample dataset with recipes & serving sizes
├── recipe_scaling.py          # Main Python script
├── README.md                  # Project documentation

🛠️ Tech Stack
Python 3
Libraries: numpy, scikit-learn, json, re, math

🚀 How to Run
Clone this repository:
git clone https://github.com/yourusername/recipe-ingredient-prediction.git
cd recipe-ingredient-prediction


Install dependencies:
pip install -r requirements.txt


Run the script:
python recipe_scaling.py

📊 Evaluation Metrics
The models are evaluated using:
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
MAPE (Mean Absolute Percentage Error)
R² Score

📌 Sample Output
Recipe: Paneer Butter Masala | Ingredient: Paneer
Known sizes: [2, 4]
Linear: {'MAE': 10.2, 'RMSE': 12.1, 'MAPE': 6.5, 'R2': 0.93}
Proportional: {'MAE': 8.7, 'RMSE': 9.4, 'MAPE': 5.8, 'R2': 0.95}
Power Law: {'MAE': 7.9, 'RMSE': 8.6, 'MAPE': 5.1, 'R2': 0.97}

🎯 Key Learning

This project demonstrates how different scaling models can be applied to real-world problems like recipe scaling, and how their performance can be compared using statistical evaluation metrics.

🖥️ Complete Python Code
import json
import random
import numpy as np
import math
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Parse Quantities
# -----------------------------
def parse_quantity(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        numbers = re.findall(r"[\d\.]+", value)
        if numbers:
            return float(numbers[-1])
    return None

# -----------------------------
# 2. Scaling Approaches
# -----------------------------
def linear_scaling(s1, q1, s2, q2, s):
    return q1 + (q2 - q1) * (s - s1) / (s2 - s1)

def proportional_scaling(s1, q1, s2, q2, s):
    factor1 = q1 / s1
    factor2 = q2 / s2
    avg_factor = (factor1 + factor2) / 2
    return avg_factor * s

def power_law_scaling(s1, q1, s2, q2, s):
    if s1 == s2 or q1 <= 0 or q2 <= 0:
        return None
    b = math.log(q2/q1) / math.log(s2/s1)
    a = q1 / (s1 ** b)
    return a * (s ** b)

# -----------------------------
# 3. Evaluation Metrics
# -----------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

# -----------------------------
# 4. Evaluation Loop
# -----------------------------
def run_evaluation(recipes):
    results = []
    for recipe, servings in recipes.items():
        serving_sizes = list(map(int, servings.keys()))
        if len(serving_sizes) < 2:
            continue

        s_known = random.sample(serving_sizes, 2)
        s_unknown = [s for s in serving_sizes if s not in s_known]

        for ingredient in servings[str(serving_sizes[0])].keys():
            q1 = parse_quantity(servings[str(s_known[0])][ingredient])
            q2 = parse_quantity(servings[str(s_known[1])][ingredient])
            if q1 is None or q2 is None:
                continue

            y_true = []
            y_pred_linear, y_pred_prop, y_pred_power = [], [], []

            for s in s_unknown:
                true_val = parse_quantity(servings[str(s)][ingredient])
                pred_lin = linear_scaling(s_known[0], q1, s_known[1], q2, s)
                pred_prop = proportional_scaling(s_known[0], q1, s_known[1], q2, s)
                pred_power = power_law_scaling(s_known[0], q1, s_known[1], q2, s)

                if pred_power is None:
                    continue

                y_true.append(true_val)
                y_pred_linear.append(pred_lin)
                y_pred_prop.append(pred_prop)
                y_pred_power.append(pred_power)

            if y_true and y_pred_linear and y_pred_prop and y_pred_power:
                metrics_lin = evaluate(y_true, y_pred_linear)
                metrics_prop = evaluate(y_true, y_pred_prop)
                metrics_power = evaluate(y_true, y_pred_power)
                results.append({
                    "recipe": recipe,
                    "ingredient": ingredient,
                    "known_sizes": s_known,
                    "linear_metrics": metrics_lin,
                    "proportional_metrics": metrics_prop,
                    "powerlaw_metrics": metrics_power
                })
    return results

# -----------------------------
# 5. Run and Report
# -----------------------------
if __name__ == "__main__":
    with open("paneer_recipes.json", "r") as f:
        recipes = json.load(f)

    final_results = run_evaluation(recipes)
    for res in final_results[:5]:
        print(f"\nRecipe: {res['recipe']} | Ingredient: {res['ingredient']}")
        print(f"Known sizes: {res['known_sizes']}")
        print("Linear:", res["linear_metrics"])
        print("Proportional:", res["proportional_metrics"])
        print("Power Law:", res["powerlaw_metrics"])

Author: Nandini
