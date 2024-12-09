import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import joblib

# determining file path
current_directory = os.path.dirname(os.path.abspath(__file__))
filename_path = os.path.join(current_directory, 'Government Efficiency Data.csv')

# load of dataset
try:
    df = pd.read_csv(filename_path)
    print("Dataset loaded successfully")
except FileNotFoundError:
    print("Error: File not found. Please check the file name and path.")
    exit()

# data exploration
print("DATASET PREVIEW:")
print(df.head())
print()

print("DATASET INFO:")
print(df.info())
print()

# data cleaning & dropping
print("CLEANING THE DATA")
df_subset = df[['Unnamed: 0', 'Unnamed: 2', 'Unnamed: 6', 'Unnamed: 10',
                'Unnamed: 14', 'Unnamed: 18']]
df_subset = df_subset.drop(axis=0, index=0)
print(df_subset)
print()

# renaming columns
df_subset = df_subset.rename(columns={'Unnamed: 0': 'Categories', 'Unnamed: 2': '2020', 'Unnamed: 6': '2021',
                                      'Unnamed: 10': '2022', 'Unnamed: 14': '2023', 'Unnamed: 18': '2024', })

# Convert columns to numeric, handling errors gracefully
numeric_cols = ['2020', '2021', '2022', '2023', '2024']
for col in numeric_cols:
    df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')

print("UPDATED DATASET")
print(df_subset)
print()
print("DESCRIPTIVE STATISTICS:")
print(df_subset.describe())
print()
print("DATA TYPES:")
print(df_subset.dtypes)
print()
print("OUTLIERS")
print(df_subset.isna().sum())
print()

# data analysis

# Are there categories with consistent top performance over the years?
# Which category showed the most improvement in score from 2020 to 2024?
# How has the Government Efficiency Overall Score changed from 2020 to 2024?

# Filter out "Government Efficiency Overall Score"
filtered_df = df_subset[df_subset["Categories"] != "Government Efficiency Overall Score"].copy()

# Consistent top performance
consistent_top = filtered_df.loc[
    (filtered_df["2020"] >= 1.5) &
    (filtered_df["2021"] >= 1.5) &
    (filtered_df["2022"] >= 1.5) &
    (filtered_df["2023"] >= 1.5) &
    (filtered_df["2024"] >= 1.5),
    "Categories"
]
print("Consistent Top Performers:", consistent_top.values)

# Most improvement in score
filtered_df["% Change (2020-2024)"] = (
                                              (filtered_df["2024"] - filtered_df["2020"]) / filtered_df["2020"]
                                      ) * 100
most_improved = filtered_df.loc[filtered_df["% Change (2020-2024)"].idxmax()]
print("\nCategory with Most Improvement:")
print(most_improved["Categories"], "with a % change of", round(most_improved["% Change (2020-2024)"], 2), "%")

# Government Efficiency Overall Score trend
gov_efficiency = df_subset.loc[df_subset["Categories"] == "Government Efficiency Overall Score",
["2020", "2021", "2022", "2023", "2024"]]
print("\nGovernment Efficiency Overall Score Trend:")
print(gov_efficiency.T)
print()

# Data Visualization

# Visualization 1: Average Scores for Top Performers (Bar Chart)
consistent_categories = consistent_top.values
top_df = filtered_df[filtered_df["Categories"].isin(consistent_categories)]
top_df = top_df.copy()
top_df["Average Score"] = top_df[["2020", "2021", "2022", "2023", "2024"]].mean(axis=1)

# Plotting the bar chart for average scores
plt.figure(figsize=(10, 6))
sns.barplot(x="Categories", y="Average Score", data=top_df, palette="viridis", hue="Categories", legend=False)
plt.title("Average Scores of Consistent Top Performers")
plt.xlabel("Categories")
plt.ylabel("Average Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Visualization 2: Percentage Improvement from 2020 to 2024 (Bar Chart)
filtered_df["% Change (2020-2024)"] = (filtered_df["2024"] - filtered_df["2020"]) / filtered_df["2020"] * 100
plt.figure(figsize=(10, 6))
sns.barplot(x="Categories", y="% Change (2020-2024)", data=filtered_df, palette="coolwarm", hue="Categories",
            legend=False)
plt.title("Percentage Improvement in Scores (2020 to 2024)")
plt.xlabel("Categories")
plt.ylabel("Percentage Change (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Visualization 3: Government Efficiency Overall Score Trend (Line Graph)
gov_efficiency_trend = gov_efficiency.T.reset_index()
gov_efficiency_trend.columns = ["Year", "Score"]
gov_efficiency_trend["Year"] = ["2020", "2021", "2022", "2023", "2024"]

plt.figure(figsize=(10, 6))
sns.lineplot(x="Year", y="Score", data=gov_efficiency_trend, marker="o", color="blue")
plt.title("Trend of Government Efficiency Overall Score (2020 to 2024)")
plt.xlabel("Year")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Dropping the 'Categories' and '% Change' columns to focus on year-to-year correlations
scores_df = filtered_df.drop(columns=["Categories", "% Change (2020-2024)"])

# Calculating the correlation matrix
correlation_matrix = scores_df.corr()

# Plotting the heatmap for correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap of Correlations Among Years")
plt.show()

# Handle NaN values (drop or impute)
df_subset = df_subset.dropna()

# machine learning
features = df_subset[['2020', '2021', '2022', '2023']]  # Features from previous years
target = df_subset['2024']  # Target: Government Efficiency Score for 2024

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Neural Network (MLP) Model
mlp = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

# Stacking Regressor Model (Using LR and MLP as base models)
stacking_model = StackingRegressor(
    estimators=[('lr', lr_model), ('mlp', mlp)],
    final_estimator=LinearRegression()
)
stacking_model.fit(X_train, y_train)
y_pred_stacking = stacking_model.predict(X_test)
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
r2_stacking = r2_score(y_test, y_pred_stacking)

# Output MSE and R2 for each model
print("Linear Regression MSE:", mse_lr, "R2:", r2_lr)
print("Neural Network MSE:", mse_mlp, "R2:", r2_mlp)
print("Stacking Regressor MSE:", mse_stacking, "R2:", r2_stacking)

# Visualizations
# Visualization 1: Comparison of MSE for all models
models = ['Linear Regression', 'Neural Network', 'Stacking Regressor']
mse_values = [mse_lr, mse_mlp, mse_stacking]

plt.figure(figsize=(8, 6))
sns.barplot(x=models, y=mse_values, palette='viridis', hue=models)  # Setting hue to x
plt.title('Comparison of Model Mean Squared Errors')
plt.ylabel('Mean Squared Error')
plt.xlabel('Model')
plt.tight_layout()
plt.show()

# Visualization 2: R-squared Comparison
r2_values = [r2_lr, r2_mlp, r2_stacking]

plt.figure(figsize=(8, 6))
sns.barplot(x=models, y=r2_values, palette='coolwarm', hue=models)  # Setting hue to x
plt.title('Comparison of Model R-squared Values')
plt.ylabel('R-squared')
plt.xlabel('Model')
plt.tight_layout()
plt.show()

# model delployment
# Save models with joblib
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(mlp, 'mlp.pkl')
joblib.dump(stacking_model, 'stacking_model.pkl')
