# 2082/01/05 01:30 AM
# This is the new version of code .. in which i have taken data in 1000m resolution
# 2082/04/22 08:01 AM
# Revised on August 10, 2025: Removed Year as feature; ANN with 2 and 3 hidden layers; Extended palette to avoid color repetition.

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle

# Define the years to process
years = [2000, 2005, 2010, 2015, 2022]

# Define base paths
base_dir = r"D:\01_Project_ML\02_ANN_and_SVR_Also"
input_dir = os.path.join(base_dir, "01_Compiled_Data")
output_base_dir = os.path.join(base_dir, "00_ALl_Models_Output")
modified_output_base_dir = os.path.join(base_dir, "02_ML_Results_for_Premonsoon")
os.makedirs(output_base_dir, exist_ok=True)
os.makedirs(modified_output_base_dir, exist_ok=True)

# A4 landscape aspect ratio
A4_LANDSCAPE_ASPECT = 11.69 / 8.27

# Initialize label font size
label_font_size = 16  # As per original code

# Function to sanitize filenames and feature names
def sanitize_name(name, for_filename=True):
    print(f"Sanitizing name: {name}")
    sanitized = name.replace('[', '').replace(']', '').replace('/', '_').replace(' ', '_')
    if not for_filename:
        sanitized = sanitized.replace('__', '_').strip('_')
    print(f"Sanitized to: {sanitized}")
    return sanitized

# Function to get readable labels
def get_readable_label(sanitized_name):
    sanitized_name = sanitized_name.replace('Lulc', 'LULC').replace('lulc', 'LULC')
    label_map = {
        "SPI": "Standardized Precipitation Index",
        "TWI": "Topographic Wetness Index",
        "Drainage_Density": "Drainage Density",
        "LULC_Forest": "LULC [Forest]",
        "LULC_Agricultural_Land": "LULC [Agricultural Land]",
        "LULC_Builtup_Area": "LULC [Builtup Area]",
        "LULC_Waterbodies": "LULC [Waterbodies]",
        "LULC_Grassland": "LULC [Grassland]",
        "LULC_Riverbed_Barren_Land": "LULC [Riverbed/Barren Land]",
        "Soil_Type_Sandy_Loam": "Soil Type [Sandy Loam]",
        "Soil_Type_Clay_Loam": "Soil Type [Clay Loam]",
        "Soil_Type_Silty_Clay_Loam": "Soil Type [Silty Clay Loam]",
        "Soil_Type_Loam": "Soil Type [Loam]",
        "Soil_Type_Sandy_Clay_Loam": "Soil Type [Sandy Clay Loam]",
        "Average_Depth": "Average Groundwater Depth",
        "OOB_Score": "Out-of-Bag Score",
        "Cross_Validation_R2": "Cross-Validation R²",
        "Points_No": "Points No",
        "Pre_monsoon_Depth": "Pre-monsoon Depth",
        "Post_monsoon_Depth": "Post-monsoon Depth"
    }
    readable = label_map.get(sanitized_name, sanitized_name.replace('_', ' ').title())
    print(f"Converted label '{sanitized_name}' to '{readable}'")
    return readable

# Function to create directories
def create_output_dirs(model_name, config, base_dir):
    print(f"Creating output directories for {model_name}, config={config}")
    output_dir = os.path.join(base_dir, model_name, f"config {config}")
    csv_output_dir = os.path.join(output_dir, "CSVs")
    plot_base_dir = os.path.join(output_dir, "Plots")
    absolute_depth_dir = os.path.join(plot_base_dir, "Absolute_Depth_Plots")
    relative_depth_dir = os.path.join(plot_base_dir, "Relative_Depth_Plots")
    significance_output_dir = os.path.join(absolute_depth_dir, "Significance")
    partial_dependence_output_dir = os.path.join(absolute_depth_dir, "Partial_Dependence")
    scatter_output_dir = os.path.join(absolute_depth_dir, "Scatter")
    pie_chart_output_dir = os.path.join(absolute_depth_dir, "Pie_Charts")
    residual_output_dir = os.path.join(relative_depth_dir, "Residuals")
    shap_output_dir = os.path.join(absolute_depth_dir, "SHAP")
    for dir_path in [csv_output_dir, plot_base_dir, absolute_depth_dir, relative_depth_dir, significance_output_dir, 
                     partial_dependence_output_dir, scatter_output_dir, pie_chart_output_dir, residual_output_dir, shap_output_dir]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    return (csv_output_dir, significance_output_dir, partial_dependence_output_dir, scatter_output_dir, 
            pie_chart_output_dir, residual_output_dir, shap_output_dir, absolute_depth_dir, relative_depth_dir)

# Step 1: Load the merged data
print("Step 1: Loading Encoded Data for All Years...")
start_time = time.time()
input_file = os.path.join(input_dir, "compiled_data_all_years_encoded.csv")
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")
combined_df = pd.read_csv(input_file, sep=',')
print(f"Loaded Data from {input_file}: {len(combined_df)} Rows")

# Check for duplicate columns
if combined_df.columns.duplicated().any():
    print(f"Warning: Duplicate columns found in {input_file}: {combined_df.columns[combined_df.columns.duplicated()].tolist()}")
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

# Sampling Options
# Option 1: Sample 500 rows for testing (active)
print("Sampling 500 rows from the dataset...")
combined_df = combined_df.sample(n=500, random_state=42)
print(f"Total rows after sampling (500 rows): {len(combined_df)}")

# # Option 2: Use the entire dataset (commented out)
# print("Using the entire dataset...")
# print(f"Total rows in the dataset: {len(combined_df)}")

# Compute dataset statistics
print("Computing dataset statistics...")
data_stats = combined_df.describe().loc[['mean', 'std', 'min', 'max']].round(4)
data_stats_df = data_stats.T.reset_index().rename(columns={'index': 'Variable'})
data_stats_df['Stage'] = 'Full Dataset'

# Outlier detection
print("Removing outliers from Pre-monsoon Depth and SPI...")
Q1 = combined_df[["Pre-monsoon Depth", "SPI"]].quantile(0.25)
Q3 = combined_df[["Pre-monsoon Depth", "SPI"]].quantile(0.75)
IQR = Q3 - Q1
combined_df = combined_df[~((combined_df[["Pre-monsoon Depth", "SPI"]] < (Q1 - 1.5 * IQR)) | 
                           (combined_df[["Pre-monsoon Depth", "SPI"]] > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f"Rows after outlier removal: {len(combined_df)}")

# Define target and feature columns
target_col = ["Pre-monsoon Depth"]
excluded_cols = target_col + ["Points No", "X", "Y", "Average Depth", "Post-monsoon Depth", "Year"]  # Removed Year
original_feature_cols = [col for col in combined_df.columns if col not in excluded_cols]

# Remove duplicates
original_feature_cols = list(dict.fromkeys(original_feature_cols))
print(f"Original features: {original_feature_cols}")

# Sanitize feature names for model training
feature_cols = [sanitize_name(col, for_filename=False) for col in original_feature_cols]
# Map original to sanitized names
feature_name_map = dict(zip(feature_cols, original_feature_cols))
# Verify no duplicates
if len(feature_cols) != len(set(feature_cols)):
    raise ValueError(f"Duplicate features detected: {feature_cols}")
print(f"Sanitized features for training: {feature_cols}")

lulc_features = [col for col in feature_cols if col.startswith("LULC_")]
soil_features = [col for col in feature_cols if col.startswith("Soil_Type_")]
# Rename columns in DataFrame for training
combined_df_sanitized = combined_df.rename(columns={orig: sanitized for orig, sanitized in zip(original_feature_cols, feature_cols)})
X = combined_df_sanitized[feature_cols].astype(float)
y = combined_df["Pre-monsoon Depth"].astype(float)

# Standardize features for all models
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=feature_cols)  # Convert back to DataFrame for consistency

# Extended color palette to avoid repetition
custom_palette = sns.color_palette("husl", 10)

# Set global plotting style as per original code
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = label_font_size
plt.rcParams['axes.labelsize'] = label_font_size
plt.rcParams['axes.titlesize'] = label_font_size
plt.rcParams['xtick.labelsize'] = label_font_size
plt.rcParams['ytick.labelsize'] = label_font_size
plt.rcParams['legend.fontsize'] = label_font_size

# Target distribution plot
print("Generating target distribution plot...")
plt.figure(figsize=(11.69, 8.27))
target_dist_data = pd.DataFrame({"Pre-monsoon Depth": combined_df["Pre-monsoon Depth"]})
hist_data, bins = np.histogram(combined_df["Pre-monsoon Depth"], bins=30)
hist_df = pd.DataFrame({"Bin Start": bins[:-1], "Bin End": bins[1:], "Frequency": hist_data})
hist_df.to_csv(os.path.join(output_base_dir, "Target_Distribution_Data.csv"), index=False)
plt.hist(combined_df["Pre-monsoon Depth"], bins=30, alpha=0.5, color=custom_palette[0], label="Full Dataset")
plt.title("Distribution of Pre-monsoon Groundwater Depth (Full Dataset)", fontsize=label_font_size)
plt.xlabel("Pre-monsoon Groundwater Depth (meters)", fontsize=label_font_size)
plt.ylabel("Frequency", fontsize=label_font_size)
legend = plt.legend(fontsize=label_font_size, frameon=True)
legend.get_frame().set_facecolor('lightgray')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.5)
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.tight_layout()
target_dist_path = os.path.join(output_base_dir, "Target_Distribution_Full.png")
plt.savefig(target_dist_path, dpi=300)
plt.close()
print(f"Saved target distribution plot: {target_dist_path}")

# Correlation heatmaps
print("Generating correlation heatmap variations...")
feature_abbreviations = {
    "Drainage Density": "DD",
    "SPI": "SPI",
    "TWI": "TWI",
    "LULC [Waterbodies]": "LULC_W",
    "LULC [Forest]": "LULC_F",
    "LULC [Riverbed/Barren Land]": "LULC_R",
    "LULC [Builtup Area]": "LULC_B",
    "LULC [Agricultural Land]": "LULC_A",
    "LULC [Grassland]": "LULC_G",
    "Soil Type [Clay Loam]": "ST_CL",
    "Soil Type [Silty Clay Loam]": "ST_SCL",
    "Soil Type [Sandy Clay Loam]": "ST_SACL",
    "Soil Type [Loam]": "ST_L",
    "Soil Type [Sandy Loam]": "ST_SL",
    "Pre-monsoon Depth": "PMD"
}
corr_cols = [col for col in original_feature_cols] + ["Pre-monsoon Depth"]
if not all(col in combined_df.columns for col in corr_cols):
    missing_cols = [col for col in corr_cols if col not in combined_df.columns]
    raise KeyError(f"Columns {missing_cols} not found in combined_df")
corr_matrix = combined_df[corr_cols].corr()
display_names = [feature_abbreviations.get(col, col) for col in corr_cols]

# Variation 1: Standard with Annotations
print("Generating Variation 1: Standard with Annotations...")
plt.figure(figsize=(11.69, 8.27))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
            annot_kws={'size': label_font_size-2}, cbar_kws={'label': 'Corr. Coeff.', 'orientation': 'horizontal', 'pad': 0.2})
plt.title("Correlation Heatmap of Features and Target", fontsize=label_font_size)
plt.xticks(ticks=np.arange(len(display_names))+0.5, labels=display_names, fontsize=label_font_size-4, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(display_names))+0.5, labels=display_names, fontsize=label_font_size-4, rotation=0)
plt.tight_layout()
corr_heatmap_path = os.path.join(output_base_dir, "Correlation_Heatmap_Var1_Annotated.png")
plt.savefig(corr_heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved Variation 1 (Annotated): {corr_heatmap_path}")

# Variation 2: Color Fill Only
print("Generating Variation 2: Color Fill Only...")
plt.figure(figsize=(11.69, 8.27))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
            cbar_kws={'label': 'Corr. Coeff.', 'orientation': 'horizontal', 'pad': 0.2})
plt.title("Correlation Heatmap of Features and Target", fontsize=label_font_size)
plt.xticks(ticks=np.arange(len(display_names))+0.5, labels=display_names, fontsize=label_font_size-4, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(display_names))+0.5, labels=display_names, fontsize=label_font_size-4, rotation=0)
plt.tight_layout()
corr_heatmap_path = os.path.join(output_base_dir, "Correlation_Heatmap_Var2_ColorOnly.png")
plt.savefig(corr_heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved Variation 2 (Color Only): {corr_heatmap_path}")

for var, cmap, size, annot, rot_x, rot_y, suffix in [
    ("Var3", "YlOrRd", (12, 9), True, 90, 0, "Annotated"),
    ("Var3", "YlOrRd", (12, 9), False, 90, 0, "ColorOnly"),
    ("Var4", "RdBu_r", (10, 10), True, 45, 0, "Annotated"),
    ("Var4", "RdBu_r", (10, 10), False, 45, 0, "ColorOnly"),
    ("Var5", "viridis", (14, 7), True, 0, 0, "Annotated"),
    ("Var5", "viridis", (14, 7), False, 0, 0, "ColorOnly"),
    ("Var6", "seismic", (9, 9), True, 45, 45, "Annotated"),
    ("Var6", "seismic", (9, 9), False, 45, 45, "ColorOnly")
]:
    print(f"Generating {var}: {'Annotated' if annot else 'Color Only'}...")
    plt.figure(figsize=size)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, center=0, fmt='.2f', 
                annot_kws={'size': label_font_size-4 if annot else None}, 
                cbar_kws={'label': 'Corr. Coeff.', 'orientation': 'horizontal', 'pad': 0.2})
    plt.title("Correlation Heatmap of Features and Target", fontsize=label_font_size)
    plt.xticks(ticks=np.arange(len(display_names))+0.5, labels=display_names, fontsize=label_font_size-5, rotation=rot_x, ha='right' if rot_x!=0 else 'center')
    plt.yticks(ticks=np.arange(len(display_names))+0.5, labels=display_names, fontsize=label_font_size-5, rotation=rot_y)
    plt.tight_layout()
    corr_heatmap_path = os.path.join(output_base_dir, f"Correlation_Heatmap_{var}_{suffix}.png")
    plt.savefig(corr_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {var} ({suffix}): {corr_heatmap_path}")

# Function to run model and save results
def run_model(model_name, config, X, y, feature_cols, lulc_features, soil_features, base_dir):
    output_dirs = create_output_dirs(model_name, config, base_dir)
    csv_output_dir, significance_output_dir, partial_dependence_output_dir, scatter_output_dir, pie_chart_output_dir, residual_output_dir, shap_output_dir, abs_depth_dir, rel_depth_dir = output_dirs
    
    print(f"Starting {model_name} training with {config}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Training set size: {len(X_train)} rows, Test set size: {len(X_test)} rows")
    
    # Initialize model
    if model_name == "RandomForest":
        model = RandomForestRegressor(n_estimators=int(config), random_state=42, n_jobs=4, oob_score=True)
    elif model_name == "XGBoost":
        model = xgb.XGBRegressor(n_estimators=int(config), random_state=42, n_jobs=4)
    elif model_name == "GradientBoosting":
        model = GradientBoostingRegressor(n_estimators=int(config), random_state=42)
    elif model_name == "ANN":
        if config == "2_layers":
            hidden_sizes = (100, 50)
        elif config == "3_layers":
            hidden_sizes = (100, 50, 25)
        model = MLPRegressor(hidden_layer_sizes=hidden_sizes, max_iter=1000, random_state=42, early_stopping=True)
    elif model_name == "SVR":
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    
    # Train model
    model.fit(X_train, y_train)
    print(f"{model_name} training completed.")
    
    # Predictions and metrics
    print("Calculating test predictions and metrics...")
    y_pred = model.predict(X_test)
    mse = round(mean_squared_error(y_test, y_pred), 4)
    rmse = round(np.sqrt(mse), 4)
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)
    oob_score = round(model.oob_score_, 4) if model_name == "RandomForest" else None
    print("Test metrics calculated.")
    
    # Cross-validation
    print("Performing 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean_r2 = round(cv_r2_scores.mean(), 4)
    cv_std_r2 = round(cv_r2_scores.std() * 2, 4)
    cv_predictions = []
    cv_actuals = []
    cv_fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        print(f"Processing CV fold {fold}/5...")
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_cv_train, y_cv_train)
        cv_pred = model.predict(X_cv_val)
        fold_mse = round(mean_squared_error(y_cv_val, cv_pred), 4)
        fold_rmse = round(np.sqrt(fold_mse), 4)
        fold_mae = round(mean_absolute_error(y_cv_val, cv_pred), 4)
        fold_r2 = round(r2_score(y_cv_val, cv_pred), 4)
        cv_fold_results.append({
            "Fold": fold,
            "MSE": fold_mse,
            "RMSE": fold_rmse,
            "MAE": fold_mae,
            "R²": fold_r2
        })
        cv_predictions.extend(cv_pred)
        cv_actuals.extend(y_cv_val)
    cv_mse = round(mean_squared_error(cv_actuals, cv_predictions), 4)
    cv_rmse = round(np.sqrt(cv_mse), 4)
    cv_mae = round(mean_absolute_error(cv_actuals, cv_predictions), 4)
    print("Cross-validation completed.")
    
    # Save CV results
    cv_fold_df = pd.DataFrame(cv_fold_results)
    cv_fold_df["config"] = config
    cv_path = os.path.join(csv_output_dir, "CV_Detailed_Results.csv")
    cv_fold_df.to_csv(cv_path, index=False)
    print(f"Saved CV results: {cv_path}")
    
    # Metrics DataFrame
    metrics_df = pd.DataFrame([{
        "Target": "Pre-monsoon Depth",
        "MSE (Test)": mse,
        "RMSE (Test)": rmse,
        "MAE (Test)": mae,
        "R² (Test)": r2,
        "OOB Score": oob_score,
        "MSE (CV)": cv_mse,
        "RMSE (CV)": cv_rmse,
        "MAE (CV)": cv_mae,
        "Mean CV R²": cv_mean_r2,
        "CV Std Dev": cv_std_r2
    }])
    metrics_df["config"] = config
    metrics_path = os.path.join(csv_output_dir, "Model Metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")
    
    # Predictions
    pred_df = pd.DataFrame({"Actual": y_test.round(4), "Predicted": y_pred.round(4)})
    pred_path = os.path.join(csv_output_dir, "Predictions Pre-monsoon Depth.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions: {pred_path}")
    cv_pred_df = pd.DataFrame({"Actual CV": cv_actuals, "Predicted CV": cv_predictions})
    cv_pred_path = os.path.join(csv_output_dir, "CV Predictions Pre-monsoon Depth.csv")
    cv_pred_df.to_csv(cv_pred_path, index=False)
    print(f"Saved CV predictions: {cv_pred_path}")
    
    # Parameter importance
    print("Calculating parameter importance...")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=4)
    importance_df = pd.DataFrame({
        "Feature": [feature_name_map[col] for col in feature_cols],
        "Importance": [round(abs(imp), 4) for imp in perm_importance.importances_mean]
    }).sort_values(by="Importance", ascending=False)
    importance_df["config"] = config
    importance_path = os.path.join(csv_output_dir, "Parameter Importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved parameter importance: {importance_path}")

    # Category importance
    lulc_importance = round(importance_df[importance_df["Feature"].isin([feature_name_map[col] for col in lulc_features])]["Importance"].sum(), 4)
    soil_importance = round(importance_df[importance_df["Feature"].isin([feature_name_map[col] for col in soil_features])]["Importance"].sum(), 4)
    category_importance = pd.DataFrame({
        "Category": ["LULC", "Soil Type"],
        "Importance": [lulc_importance, soil_importance]
    })
    category_importance["config"] = config
    cat_path = os.path.join(csv_output_dir, "Category Importance.csv")
    category_importance.to_csv(cat_path, index=False)
    print(f"Saved category importance: {cat_path}")

    # Aggregated parameters (Top 5: Top 3 individual + LULC + Soil Type)
    top_individual_params = importance_df[~importance_df["Feature"].isin([feature_name_map[col] for col in lulc_features + soil_features])].nlargest(3, "Importance")
    aggregated_params = pd.concat([
        top_individual_params,
        pd.DataFrame({"Feature": ["LULC", "Soil Type"], "Importance": [lulc_importance, soil_importance]})
    ]).sort_values(by="Importance", ascending=False)

    # Plotting setup
    plt.rcParams['font.family'] = 'Times New Roman'

    # Plot 1: Parameter Importance (All Sub-Parameters)
    print(f"Generating Parameters vs Relative Importance (All Sub-Parameters) plot for {model_name} with {config}...")
    plt.figure(figsize=(11.69, 8.27))
    sns.barplot(x="Importance", y="Feature", data=importance_df, width=0.5, hue="Feature", palette=custom_palette, edgecolor='black', linewidth=0.5, legend=False)
    max_imp = importance_df["Importance"].max()
    plt.xlim(0, max_imp * 1.3)
    plt.xticks(ticks=np.arange(0, max_imp * 1.3, 0.1), labels=[f"{int(tick * 100)}%" for tick in np.arange(0, max_imp * 1.3, 0.1)])
    for i, v in enumerate(importance_df["Importance"]):
        plt.text(v + max_imp * 0.01, i, f"{v*100:.2f}%", va="center", ha="left", fontsize=label_font_size)
    plt.title(f"Parameters vs Relative Importance (All Sub-Parameters) ({model_name}, {config})", fontsize=label_font_size)
    plt.xlabel("Relative Importance (%)", fontsize=label_font_size)
    plt.ylabel("Parameters", fontsize=label_font_size)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    all_params_plot_path = os.path.join(significance_output_dir, "Parameters vs Relative Importance (All Sub-Parameters).png")
    plt.savefig(all_params_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {all_params_plot_path}")

    # Plot 2: Aggregated Importance (Top 5 Parameters)
    print(f"Generating Aggregated Importance (Top 5 Parameters) plot for {model_name} with {config}...")
    plt.figure(figsize=(11.69, 8.27))
    sns.barplot(x="Importance", y="Feature", data=aggregated_params, width=0.5, hue="Feature", palette=custom_palette[:len(aggregated_params)], edgecolor='black', linewidth=0.5, legend=False)
    max_agg_imp = aggregated_params["Importance"].max()
    plt.xlim(0, max_agg_imp * 1.3)
    plt.xticks(ticks=np.arange(0, max_agg_imp * 1.3, 0.1), labels=[f"{int(tick * 100)}%" for tick in np.arange(0, max_agg_imp * 1.3, 0.1)])
    for i, v in enumerate(aggregated_params["Importance"]):
        plt.text(v + max_agg_imp * 0.01, i, f"{v*100:.2f}%", va="center", ha="left", fontsize=label_font_size)
    plt.title(f"Parameters vs Relative Importance (Top 5 Parameters) ({model_name}, {config})", fontsize=label_font_size)
    plt.xlabel("Relative Importance (%)", fontsize=label_font_size)
    plt.ylabel("Parameters", fontsize=label_font_size)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    agg_plot_path = os.path.join(significance_output_dir, "Parameters vs Relative Importance (Top 5 Parameters).png")
    plt.savefig(agg_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {agg_plot_path}")

    # Plot 3: Category Importance
    print(f"Generating Category Importance plot for {model_name} with {config}...")
    plt.figure(figsize=(11.69, 8.27))
    sns.barplot(x="Importance", y="Category", data=category_importance, width=0.5, hue="Category", palette=custom_palette[:2], edgecolor='black', linewidth=0.5, legend=False)
    max_cat_imp = category_importance["Importance"].max()
    plt.xlim(0, max_cat_imp * 1.3)
    plt.xticks(ticks=np.arange(0, max_cat_imp * 1.3, 0.05), labels=[f"{int(tick * 100)}%" for tick in np.arange(0, max_cat_imp * 1.3, 0.05)])
    for i, v in enumerate(category_importance["Importance"]):
        plt.text(v + max_cat_imp * 0.01, i, f"{v*100:.2f}%", va="center", ha="left", fontsize=label_font_size)
    plt.title(f"Category Importance ({model_name}, {config})", fontsize=label_font_size)
    plt.xlabel("Relative Importance (%)", fontsize=label_font_size)
    plt.ylabel("Categories", fontsize=label_font_size)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    cat_plot_path = os.path.join(significance_output_dir, "Category Importance.png")
    plt.savefig(cat_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {cat_plot_path}")

    # Plot 4: LULC-specific significance
    lulc_specific_df = importance_df[importance_df["Feature"].isin([feature_name_map[col] for col in lulc_features])].copy()
    if not lulc_specific_df.empty:
        lulc_specific_df["Importance"] = (lulc_specific_df["Importance"] / lulc_importance * 100).round(2)
        lulc_path = os.path.join(csv_output_dir, "LULC Specific Importance.csv")
        lulc_specific_df.to_csv(lulc_path, index=False)
        print(f"Saved LULC-specific importance: {lulc_path}")
        plt.figure(figsize=(11.69, 8.27))
        sns.barplot(x="Importance", y="Feature", data=lulc_specific_df, width=0.5, hue="Feature", palette=custom_palette[:len(lulc_specific_df)], edgecolor='black', linewidth=0.5, legend=False)
        max_lulc_imp = lulc_specific_df["Importance"].max()
        plt.xlim(0, max_lulc_imp * 1.3)
        plt.xticks(ticks=np.arange(0, max_lulc_imp * 1.3, 10), labels=[f"{int(tick)}%" for tick in np.arange(0, max_lulc_imp * 1.3, 10)])
        for i, v in enumerate(lulc_specific_df["Importance"]):
            plt.text(v + max_lulc_imp * 0.01, i, f"{v:.2f}%", va="center", ha="left", fontsize=label_font_size)
        plt.title(f"Relative Significance Within LULC Categories ({model_name}, {config})", fontsize=label_font_size)
        plt.xlabel("Relative Significance (%)", fontsize=label_font_size)
        plt.ylabel("LULC Types", fontsize=label_font_size)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        lulc_plot_path = os.path.join(significance_output_dir, "LULC Significance.png")
        plt.savefig(lulc_plot_path, dpi=300)
        plt.close()
        print(f"Saved LULC plot: {lulc_plot_path}")

    # Plot 5: Soil Type-specific significance
    soil_specific_df = importance_df[importance_df["Feature"].isin([feature_name_map[col] for col in soil_features])].copy()
    if not soil_specific_df.empty:
        soil_specific_df["Importance"] = (soil_specific_df["Importance"] / soil_importance * 100).round(2)
        soil_path = os.path.join(csv_output_dir, "Soil Type Specific Importance.csv")
        soil_specific_df.to_csv(soil_path, index=False)
        print(f"Saved Soil Type-specific importance: {soil_path}")
        plt.figure(figsize=(11.69, 8.27))
        sns.barplot(x="Importance", y="Feature", data=soil_specific_df, width=0.5, hue="Feature", palette=custom_palette[:len(soil_specific_df)], edgecolor='black', linewidth=0.5, legend=False)
        max_soil_imp = soil_specific_df["Importance"].max()
        plt.xlim(0, max_soil_imp * 1.3)
        plt.xticks(ticks=np.arange(0, max_soil_imp * 1.3, 10), labels=[f"{int(tick)}%" for tick in np.arange(0, max_soil_imp * 1.3, 10)])
        for i, v in enumerate(soil_specific_df["Importance"]):
            plt.text(v + max_soil_imp * 0.01, i, f"{v:.2f}%", va="center", ha="left", fontsize=label_font_size)
        plt.title(f"Relative Significance Within Soil Type Categories ({model_name}, {config})", fontsize=label_font_size)
        plt.xlabel("Relative Significance (%)", fontsize=label_font_size)
        plt.ylabel("Soil Types", fontsize=label_font_size)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        soil_plot_path = os.path.join(significance_output_dir, "Soil Type Significance.png")
        plt.savefig(soil_plot_path, dpi=300)
        plt.close()
        print(f"Saved Soil Type plot: {soil_plot_path}")

    # Pie Chart
    print(f"Generating Pie Chart for {model_name} with {config}...")
    plt.figure(figsize=(11.69, 8.27))
    wedges, texts, autotexts = plt.pie(importance_df["Importance"], autopct='%.2f%%', startangle=90, colors=custom_palette, textprops={'fontsize': label_font_size})
    for i, (wedge, text, autotext) in enumerate(zip(wedges, texts, autotexts)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 1.5 * np.cos(np.radians(angle))
        y = 1.5 * np.sin(np.radians(angle))
        plt.annotate(get_readable_label(importance_df["Feature"].iloc[i]), xy=(x, y), xytext=(1.7 * x, 1.7 * y), 
                     arrowprops=dict(arrowstyle="->"), fontsize=label_font_size)
        text.set_visible(False)
    plt.title(f"Distribution of Relative Importance ({model_name}, {config})", fontsize=label_font_size)
    plt.axis('equal')
    plt.tight_layout()
    pie_path = os.path.join(pie_chart_output_dir, "Pie Chart All Parameters.png")
    plt.savefig(pie_path, dpi=300)
    plt.close()
    print(f"Saved pie chart: {pie_path}")

    # Scatter plot (Actual vs Predicted for 30% test data)
    print(f"Generating Scatter plot for {model_name} with {config}...")
    plt.figure(figsize=(11.69, 8.27))
    plt.scatter(pred_df["Actual"], pred_df["Predicted"], alpha=0.5, c=custom_palette[1])
    plt.plot([pred_df["Actual"].min(), pred_df["Actual"].max()], [pred_df["Actual"].min(), pred_df["Actual"].max()], 'r--', lw=2)
    plt.title(f"Actual vs Predicted Pre-monsoon Groundwater Depth ({model_name}, {config})", fontsize=label_font_size)
    plt.xlabel("Actual Pre-monsoon Groundwater Depth (meters)", fontsize=label_font_size)
    plt.ylabel("Predicted Pre-monsoon Groundwater Depth (meters)", fontsize=label_font_size)
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    scatter_path = os.path.join(scatter_output_dir, "Actual vs Predicted.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved scatter plot: {scatter_path}")

    # Additional Scatter Plot (Actual Depth vs Predicted)
    print(f"Generating Actual Depth vs Predicted plot for {model_name} with {config}...")
    plt.figure(figsize=(11.69, 8.27))
    plt.scatter(pred_df["Actual"], pred_df["Predicted"], alpha=0.5, c=custom_palette[1])
    plt.plot([pred_df["Actual"].min(), pred_df["Actual"].max()], [pred_df["Actual"].min(), pred_df["Actual"].max()], 'r--', lw=2)
    plt.title(f"Actual Depth vs Predicted ({model_name}, {config})", fontsize=label_font_size)
    plt.xlabel("Actual Pre-monsoon Groundwater Depth (meters)", fontsize=label_font_size)
    plt.ylabel("Predicted Pre-monsoon Groundwater Depth (meters)", fontsize=label_font_size)
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    abs_depth_vs_pred_path = os.path.join(abs_depth_dir, "Actual_Depth_vs_Predicted.png")
    plt.savefig(abs_depth_vs_pred_path, dpi=300)
    plt.close()
    print(f"Saved actual depth vs predicted plot: {abs_depth_vs_pred_path}")

    # Residual plot (for 30% test data)
    print(f"Generating Residual plot for {model_name} with {config}...")
    residuals = pred_df["Actual"] - pred_df["Predicted"]
    plt.figure(figsize=(11.69, 8.27))
    plt.scatter(pred_df["Predicted"], residuals, alpha=0.5, c=custom_palette[1])
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.title(f"Residuals vs Predicted Values ({model_name}, {config})", fontsize=label_font_size)
    plt.xlabel("Predicted Pre-monsoon Groundwater Depth (meters)", fontsize=label_font_size)
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=label_font_size)
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    residual_path = os.path.join(residual_output_dir, "Residuals vs Predicted.png")
    plt.savefig(residual_path, dpi=300)
    plt.close()
    print(f"Saved residual plot: {residual_path}")

    # Partial Dependence
    print("Calculating partial dependence...")
    partial_dependence_results = []
    pd_plot_paths = []
    for feature in feature_cols:
        print(f"Processing partial dependence for {feature}...")
        pd_results = partial_dependence(model, X, [feature], kind="average")
        pd_values = pd_results["average"][0].round(4)
        pd_grid = pd_results["grid_values"][0]
        slope = np.polyfit(pd_grid, pd_values, 1)[0] if len(pd_grid) > 1 else 0
        effect_direction = "Positive" if slope > 0 else "Negative" if slope < 0 else "Neutral"
        rel_importance = importance_df[importance_df["Feature"] == feature_name_map[feature]]["Importance"].iloc[0] * 100
        partial_dependence_results.append({
            "Feature": feature_name_map[feature],
            "Effect Direction": effect_direction,
            "Average Partial Dependence": round(pd_values.mean(), 4),
            "Min Partial Dependence": round(pd_values.min(), 4),
            "Max Partial Dependence": round(pd_values.max(), 4),
            "config": config
        })
        plt.figure(figsize=(11.69, 8.27))
        plt.plot(pd_grid, pd_values, marker="o", color=custom_palette[0], 
                 label=f"Effect: {effect_direction}\nImportance: {rel_importance:.2f}%")
        plt.title(f"Partial Dependence of Pre-monsoon Groundwater Depth on {get_readable_label(feature)} ({model_name}, {config})", fontsize=label_font_size)
        plt.xlabel(get_readable_label(feature), fontsize=label_font_size)
        plt.ylabel("Pre-monsoon Groundwater Depth (meters)", fontsize=label_font_size)
        y_range = pd_values.max() - pd_values.min()
        y_mid = (pd_values.max() + pd_values.min()) / 2
        if effect_direction == "Positive":
            legend_loc = "lower right" if pd_values[-1] > y_mid else "upper right"
        elif effect_direction == "Negative":
            legend_loc = "upper right" if pd_values[-1] < y_mid else "lower right"
        else:
            legend_loc = "center right"
        legend = plt.legend(loc=legend_loc, fontsize=label_font_size, frameon=True)
        legend.get_frame().set_facecolor('lightgray')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.5)
        plt.grid(axis='both', linestyle='--', alpha=0.7)
        plt.tight_layout()
        pd_plot_path = os.path.join(partial_dependence_output_dir, f"Partial Dependence {sanitize_name(feature_name_map[feature])}.png")
        plt.savefig(pd_plot_path, dpi=300)
        plt.close()
        pd_plot_paths.append(pd_plot_path)
        print(f"Saved partial dependence plot: {pd_plot_path}")
    
    # Two-way PDP for SPI and TWI
    print(f"Generating two-way partial dependence for SPI and TWI for {model_name} with {config}...")
    features = ["SPI", "TWI"]
    pdp = partial_dependence(model, X, features, kind="average")
    plt.figure(figsize=(11.69, 8.27))
    plt.contourf(pdp["grid_values"][0], pdp["grid_values"][1], pdp["average"][0], cmap="coolwarm")
    plt.colorbar(label="Pre-monsoon Groundwater Depth (meters)")
    plt.xlabel("Standardized Precipitation Index", fontsize=label_font_size)
    plt.ylabel("Topographic Wetness Index", fontsize=label_font_size)
    plt.title(f"Two-way Partial Dependence: SPI and TWI ({model_name}, {config})", fontsize=label_font_size)
    plt.tight_layout()
    twoway_pdp_path = os.path.join(partial_dependence_output_dir, "TwoWay_PDP_SPI_TWI.png")
    plt.savefig(twoway_pdp_path, dpi=300)
    plt.close()
    print(f"Saved two-way PDP: {twoway_pdp_path}")

    # Two-way PDP for SPI and Drainage Density
    print(f"Generating two-way partial dependence for SPI and Drainage Density for {model_name} with {config}...")
    features = ["SPI", "Drainage_Density"]
    pdp = partial_dependence(model, X, features, kind="average")
    plt.figure(figsize=(11.69, 8.27))
    plt.contourf(pdp["grid_values"][0], pdp["grid_values"][1], pdp["average"][0], cmap="coolwarm")
    plt.colorbar(label="Pre-monsoon Groundwater Depth (meters)")
    plt.xlabel("Standardized Precipitation Index", fontsize=label_font_size)
    plt.ylabel("Drainage Density", fontsize=label_font_size)
    plt.title(f"Two-way Partial Dependence: SPI and Drainage Density ({model_name}, {config})", fontsize=label_font_size)
    plt.tight_layout()
    twoway_pdp_spi_dd_path = os.path.join(partial_dependence_output_dir, "TwoWay_PDP_SPI_DD.png")
    plt.savefig(twoway_pdp_spi_dd_path, dpi=300)
    plt.close()
    print(f"Saved two-way PDP: {twoway_pdp_spi_dd_path}")

    # SHAP Summary Plot with Enhancements (only for tree-based models)
    shap_path = None
    if model_name in ["RandomForest", "XGBoost", "GradientBoosting"]:
        print(f"Generating enhanced SHAP summary plot for {model_name} with {config}...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(11.69, 8.27))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=[feature_name_map[col] for col in feature_cols],
            show=False,
            plot_type="dot",
            cmap=plt.cm.coolwarm,
            max_display=len(feature_cols),
            color_bar=True
        )
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        num_features = len(feature_cols)
        for i in range(num_features):
            ax.axhline(y=i, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.7)
        plt.title(f"SHAP Summary Plot ({model_name}, {config})", fontsize=label_font_size, pad=20)
        plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=label_font_size)
        plt.ylabel("Features", fontsize=label_font_size)
        ax.tick_params(axis='x', labelsize=label_font_size-2)
        ax.tick_params(axis='y', labelsize=label_font_size-2)
        try:
            cbar = ax.collections[0].colorbar
            if cbar is not None:
                cbar.set_label('Feature Value', fontsize=label_font_size)
                cbar.ax.tick_params(labelsize=label_font_size-2)
            else:
                print("Warning: Colorbar not found in SHAP summary plot.")
        except AttributeError:
            print("Warning: Could not access colorbar for SHAP summary plot.")
        plt.tight_layout()
        shap_path = os.path.join(shap_output_dir, "SHAP_Summary.png")
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved enhanced SHAP plot: {shap_path}")

    # Save partial dependence results to CSV
    pd_df = pd.DataFrame(partial_dependence_results)
    pd_df.to_csv(os.path.join(csv_output_dir, "Partial Dependence Summary.csv"), index=False)
    print(f"Saved partial dependence summary: {os.path.join(csv_output_dir, 'Partial Dependence Summary.csv')}")
    
    return (metrics_df, importance_df, cv_fold_df, pred_df, pd_df, all_params_plot_path, agg_plot_path, pie_path, cat_plot_path, 
            lulc_plot_path, soil_plot_path, scatter_path, residual_path, shap_path, twoway_pdp_path, pd_plot_paths, abs_depth_vs_pred_path, twoway_pdp_spi_dd_path)

# Run models
models = ["RandomForest", "XGBoost", "GradientBoosting", "ANN", "SVR"]
results = {}
for model_name in models:
    print(f"\nRunning {model_name}...")
    if model_name in ["RandomForest", "XGBoost", "GradientBoosting"]:
        config_list = [100, 200]
    elif model_name == "ANN":
        config_list = ["2_layers", "3_layers"]
    else:
        config_list = ["default"]  # SVR
    for config in config_list:
        print(f"Processing {config} configuration...")
        base_dir = modified_output_base_dir
        results[(model_name, config)] = run_model(model_name, config, X, y, feature_cols, lulc_features, soil_features, base_dir)
        print(f"{model_name} with {config} configuration completed.")

# Step 2: Compare Results
print("\nStep 2: Comparing Results Across Models...")
comparison_dir = os.path.join(modified_output_base_dir, "Model_Comparisons")
os.makedirs(comparison_dir, exist_ok=True)

# Combine metrics
print("Combining metrics for all models...")
metrics_comparison = pd.concat([results[(m, c)][0].assign(Model=m) for m, c in results])
metrics_comparison.to_csv(os.path.join(comparison_dir, "Metrics_Comparison.csv"), index=False)
print(f"Saved metrics comparison: {os.path.join(comparison_dir, 'Metrics_Comparison.csv')}")

# Type 1: R² Comparison Bar Plot (All Configurations)
print("Generating R² Comparison Bar plot (All Configurations)...")
plt.figure(figsize=(11.69, 8.27))
r2_comparison = metrics_comparison[["Model", "config", "R² (Test)"]].copy()
r2_comparison["Model_config"] = r2_comparison["Model"] + "_" + r2_comparison["config"].astype(str)
sns.barplot(x="R² (Test)", y="Model_config", data=r2_comparison, palette=custom_palette[:len(r2_comparison)], edgecolor='black', linewidth=0.5)
plt.title("Comparison of R² Scores Across All Model Configurations", fontsize=label_font_size)
plt.xlabel("R² Score (Test)", fontsize=label_font_size)
plt.ylabel("Model (config)", fontsize=label_font_size)
plt.xlim(0, 1)
plt.xticks(ticks=np.arange(0, 1.1, 0.1))
for i, v in enumerate(r2_comparison["R² (Test)"]):
    plt.text(v + 0.01, i, f"{v:.2f}", va="center", ha="left", fontsize=label_font_size)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
r2_plot_path_all = os.path.join(comparison_dir, "R2_Comparison_Bar_All_Configurations.png")
plt.savefig(r2_plot_path_all, dpi=300)
plt.close()
print(f"Saved R² comparison plot (All Configurations): {r2_plot_path_all}")

# Type 2: R² Comparison Bar Plot (Averaged by Model Type)
print("Generating R² Comparison Bar plot (Averaged by Model Type)...")
r2_avg = metrics_comparison.groupby("Model")["R² (Test)"].mean().reset_index()
plt.figure(figsize=(11.69, 8.27))
sns.barplot(x="R² (Test)", y="Model", data=r2_avg, palette=custom_palette[:len(r2_avg)], edgecolor='black', linewidth=0.5)
plt.title("Comparison of Average R² Scores by Model Type", fontsize=label_font_size)
plt.xlabel("Average R² Score (Test)", fontsize=label_font_size)
plt.ylabel("Model", fontsize=label_font_size)
plt.xlim(0, 1)
plt.xticks(ticks=np.arange(0, 1.1, 0.1))
for i, v in enumerate(r2_avg["R² (Test)"]):
    plt.text(v + 0.01, i, f"{v:.2f}", va="center", ha="left", fontsize=label_font_size)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
r2_plot_path_avg = os.path.join(comparison_dir, "R2_Comparison_Bar_Averaged_Models.png")
plt.savefig(r2_plot_path_avg, dpi=300)
plt.close()
print(f"Saved R² comparison plot (Averaged by Model Type): {r2_plot_path_avg}")

# Type 1: Metrics Comparison Bar Plot (All Configurations)
print("Generating Metrics Comparison Bar plot (All Configurations)...")
plt.figure(figsize=(11.69, 8.27))
metrics_comparison["Model_config"] = metrics_comparison["Model"] + "_" + metrics_comparison["config"].astype(str)
metrics_melted = metrics_comparison.melt(id_vars=["Model_config"], 
                                         value_vars=["MSE (Test)", "RMSE (Test)", "MAE (Test)", "R² (Test)"])
sns.barplot(x="variable", y="value", hue="Model_config", data=metrics_melted, width=0.5, palette=custom_palette[:len(metrics_comparison["Model_config"].unique())])
plt.title("Comparison of Model Performance Metrics (All Configurations)", fontsize=label_font_size)
plt.xlabel("Performance Metric", fontsize=label_font_size)
plt.ylabel("Value", fontsize=label_font_size)
plt.xticks(rotation=45, fontsize=label_font_size)
legend = plt.legend(title="Model (config)", fontsize=label_font_size-2, frameon=True)
legend.get_frame().set_facecolor('lightgray')
legend.get_frame().set_edgecolor('black')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
metrics_plot_path_all = os.path.join(comparison_dir, "Metrics_Comparison_Bar_All_Configurations.png")
plt.savefig(metrics_plot_path_all, dpi=300)
plt.close()
print(f"Saved plot: {metrics_plot_path_all}")

# Type 2: Metrics Comparison Bar Plot (Averaged by Model Type)
print("Generating Metrics Comparison Bar plot (Averaged by Model Type)...")
metrics_avg = metrics_comparison.groupby("Model")[["MSE (Test)", "RMSE (Test)", "MAE (Test)", "R² (Test)"]].mean().reset_index()
plt.figure(figsize=(11.69, 8.27))
metrics_melted_avg = metrics_avg.melt(id_vars=["Model"], 
                                      value_vars=["MSE (Test)", "RMSE (Test)", "MAE (Test)", "R² (Test)"])
sns.barplot(x="variable", y="value", hue="Model", data=metrics_melted_avg, width=0.5, palette=custom_palette[:len(metrics_avg)])
plt.title("Comparison of Average Model Performance Metrics by Model Type", fontsize=label_font_size)
plt.xlabel("Performance Metric", fontsize=label_font_size)
plt.ylabel("Value", fontsize=label_font_size)
plt.xticks(rotation=45, fontsize=label_font_size)
legend = plt.legend(title="Model", fontsize=label_font_size, frameon=True)
legend.get_frame().set_facecolor('lightgray')
legend.get_frame().set_edgecolor('black')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
metrics_plot_path_avg = os.path.join(comparison_dir, "Metrics_Comparison_Bar_Averaged_Models.png")
plt.savefig(metrics_plot_path_avg, dpi=300)
plt.close()
print(f"Saved plot: {metrics_plot_path_avg}")

# Compare parameter importance
print("Combining parameter importance for all models...")
importance_comparison = pd.concat([
    results[(m, c)][1].rename(columns={"Importance": f"Importance_{m}_{c}"}).set_index("Feature")
    for m, c in results
], axis=1).reset_index()
importance_comparison.to_csv(os.path.join(comparison_dir, "Parameter_Importance_Comparison.csv"), index=False)
print(f"Saved parameter importance comparison: {os.path.join(comparison_dir, 'Parameter_Importance_Comparison.csv')}")

# Type 1: Parameter Importance Comparison Bar Plot (All Configurations)
print("Generating Parameter Importance Comparison Bar plot (All Configurations)...")
plt.figure(figsize=(11.69, 8.27))
importance_melted = importance_comparison.melt(id_vars=["Feature"], 
                                              value_vars=[f"Importance_{m}_{c}" for m, c in results],
                                              var_name="Model_config", value_name="Importance")
importance_melted["Model_config"] = importance_melted["Model_config"].str.replace("Importance_", "")
sns.barplot(x="Importance", y="Feature", hue="Model_config", data=importance_melted, width=0.5, palette=custom_palette[:len(results)])
plt.title("Comparison of Parameter Importance Across All Model Configurations", fontsize=label_font_size)
plt.xlabel("Relative Importance (%)", fontsize=label_font_size)
plt.ylabel("Parameters", fontsize=label_font_size)
max_imp_comp = importance_melted["Importance"].max()
plt.xlim(0, max_imp_comp * 1.3)
plt.xticks(ticks=np.arange(0, max_imp_comp * 1.3, 0.1), labels=[f"{int(tick * 100)}%" for tick in np.arange(0, max_imp_comp * 1.3, 0.1)])
legend = plt.legend(title="Model (config)", fontsize=label_font_size-2, frameon=True)
legend.get_frame().set_facecolor('lightgray')
legend.get_frame().set_edgecolor('black')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
importance_plot_path_all = os.path.join(comparison_dir, "Parameter_Importance_Comparison_Bar_All_Configurations.png")
plt.savefig(importance_plot_path_all, dpi=300)
plt.close()
print(f"Saved plot: {importance_plot_path_all}")

# Type 2: Parameter Importance Comparison Bar Plot (Averaged by Model Type)
print("Generating Parameter Importance Comparison Bar plot (Averaged by Model Type)...")
importance_avg = importance_comparison.copy()
importance_avg["Importance_RandomForest"] = importance_avg[["Importance_RandomForest_100", "Importance_RandomForest_200"]].mean(axis=1)
importance_avg["Importance_XGBoost"] = importance_avg[["Importance_XGBoost_100", "Importance_XGBoost_200"]].mean(axis=1)
importance_avg["Importance_GradientBoosting"] = importance_avg[["Importance_GradientBoosting_100", "Importance_GradientBoosting_200"]].mean(axis=1)
importance_avg["Importance_ANN"] = importance_avg[["Importance_ANN_2_layers", "Importance_ANN_3_layers"]].mean(axis=1)
importance_avg["Importance_SVR"] = importance_avg[["Importance_SVR_default"]].mean(axis=1)
importance_avg = importance_avg[["Feature", "Importance_RandomForest", "Importance_XGBoost", "Importance_GradientBoosting", "Importance_ANN", "Importance_SVR"]]
plt.figure(figsize=(11.69, 8.27))
importance_melted_avg = importance_avg.melt(id_vars=["Feature"], 
                                            value_vars=["Importance_RandomForest", "Importance_XGBoost", "Importance_GradientBoosting", "Importance_ANN", "Importance_SVR"],
                                            var_name="Model", value_name="Importance")
importance_melted_avg["Model"] = importance_melted_avg["Model"].str.replace("Importance_", "")
sns.barplot(x="Importance", y="Feature", hue="Model", data=importance_melted_avg, width=0.5, palette=custom_palette[:len(models)])
plt.title("Comparison of Average Parameter Importance by Model Type", fontsize=label_font_size)
plt.xlabel("Relative Importance (%)", fontsize=label_font_size)
plt.ylabel("Parameters", fontsize=label_font_size)
max_imp_comp_avg = importance_melted_avg["Importance"].max()
plt.xlim(0, max_imp_comp_avg * 1.3)
plt.xticks(ticks=np.arange(0, max_imp_comp_avg * 1.3, 0.1), labels=[f"{int(tick * 100)}%" for tick in np.arange(0, max_imp_comp_avg * 1.3, 0.1)])
legend = plt.legend(title="Model", fontsize=label_font_size, frameon=True)
legend.get_frame().set_facecolor('lightgray')
legend.get_frame().set_edgecolor('black')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
importance_plot_path_avg = os.path.join(comparison_dir, "Parameter_Importance_Comparison_Bar_Averaged_Models.png")
plt.savefig(importance_plot_path_avg, dpi=300)
plt.close()
print(f"Saved plot: {importance_plot_path_avg}")

# Type 1: Box plot for CV R² scores (All Configurations)
print("Generating CV R² Boxplot (All Configurations)...")
cv_r2_comparison = pd.concat([
    pd.DataFrame({
        "R² Score": results[(m, c)][2]["R²"],
        "Model": m,
        "config": c,
        "Model_config": f"{m}_{c}"
    }) for m, c in results
])
plt.figure(figsize=(11.69, 8.27))
sns.boxplot(x="Model_config", y="R² Score", data=cv_r2_comparison, palette=custom_palette[:len(cv_r2_comparison["Model_config"].unique())])
plt.title("Cross-Validation R² Scores Across All Model Configurations", fontsize=label_font_size)
plt.xlabel("Model (config)", fontsize=label_font_size)
plt.ylabel("R² Score", fontsize=label_font_size)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
cv_boxplot_path_all = os.path.join(comparison_dir, "CV_R2_Boxplot_All_Configurations.png")
plt.savefig(cv_boxplot_path_all, dpi=300)
plt.close()
print(f"Saved box plot: {cv_boxplot_path_all}")

# Type 2: Box plot for CV R² scores (Averaged by Model Type)
print("Generating CV R² Boxplot (Averaged by Model Type)...")
plt.figure(figsize=(11.69, 8.27))
sns.boxplot(x="Model", y="R² Score", data=cv_r2_comparison, palette=custom_palette[:len(cv_r2_comparison["Model"].unique())])
plt.title("Cross-Validation R² Scores by Model Type", fontsize=label_font_size)
plt.xlabel("Model", fontsize=label_font_size)
plt.ylabel("R² Score", fontsize=label_font_size)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
cv_boxplot_path_avg = os.path.join(comparison_dir, "CV_R2_Boxplot_Averaged_Models.png")
plt.savefig(cv_boxplot_path_avg, dpi=300)
plt.close()
print(f"Saved box plot: {cv_boxplot_path_avg}")

# Collect residuals and errors for all models
print("Collecting residuals and prediction errors for all models...")
residuals_data = []
errors_data = []
for model_name, config in results:
    pred_df = results[(model_name, config)][3]  # Predictions DataFrame
    residuals = pred_df["Actual"] - pred_df["Predicted"]
    errors = np.abs(pred_df["Actual"] - pred_df["Predicted"])
    residuals_data.append({
        "Model": model_name,
        "config": config,
        "Residual": residuals
    })
    errors_data.append({
        "Model": model_name,
        "config": config,
        "Error": errors
    })

# Type 1: Box plot for residuals (All Configurations)
print("Generating Residuals Boxplot (All Configurations)...")
residuals_df = pd.concat([
    pd.DataFrame({
        "Residual": data["Residual"],
        "Model": data["Model"],
        "config": data["config"],
        "Model_config": f"{data['Model']}_{data['config']}"
    }) for data in residuals_data
])
plt.figure(figsize=(11.69, 8.27))
sns.boxplot(x="Residual", y="Model_config", data=residuals_df, palette=custom_palette[:len(residuals_df["Model_config"].unique())])
plt.axvline(x=0, color='red', linestyle='--', lw=2)
plt.title("Residuals Distribution Across All Model Configurations", fontsize=label_font_size)
plt.xlabel("Residual (Actual - Predicted, meters)", fontsize=label_font_size)
plt.ylabel("Model (config)", fontsize=label_font_size)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
residuals_boxplot_path_all = os.path.join(comparison_dir, "Residuals_Boxplot_All_Configurations.png")
plt.savefig(residuals_boxplot_path_all, dpi=300)
plt.close()
print(f"Saved residuals box plot: {residuals_boxplot_path_all}")

# Type 2: Box plot for residuals (Averaged by Model Type)
print("Generating Residuals Boxplot (Averaged by Model Type)...")
residuals_df_avg = pd.concat([
    pd.DataFrame({
        "Residual": data["Residual"],
        "Model": data["Model"]
    }) for data in residuals_data
])
plt.figure(figsize=(11.69, 8.27))
sns.boxplot(x="Residual", y="Model", data=residuals_df_avg, palette=custom_palette[:len(residuals_df_avg["Model"].unique())])
plt.axvline(x=0, color='red', linestyle='--', lw=2)
plt.title("Residuals Distribution by Model Type", fontsize=label_font_size)
plt.xlabel("Residual (Actual - Predicted, meters)", fontsize=label_font_size)
plt.ylabel("Model", fontsize=label_font_size)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
residuals_boxplot_path_avg = os.path.join(comparison_dir, "Residuals_Boxplot_Averaged_Models.png")
plt.savefig(residuals_boxplot_path_avg, dpi=300)
plt.close()
print(f"Saved residuals box plot: {residuals_boxplot_path_avg}")

# Type 1: Violin plot for absolute prediction errors (All Configurations)
print("Generating Prediction Errors Violin Plot (All Configurations)...")
errors_df = pd.concat([
    pd.DataFrame({
        "Error": data["Error"],
        "Model": data["Model"],
        "config": data["config"],
        "Model_config": f"{data['Model']}_{data['config']}"
    }) for data in errors_data
])
plt.figure(figsize=(11.69, 8.27))
sns.violinplot(x="Error", y="Model_config", data=errors_df, palette=custom_palette[:len(errors_df["Model_config"].unique())])
plt.title("Absolute Prediction Errors Across All Model Configurations", fontsize=label_font_size)
plt.xlabel("Absolute Prediction Error (meters)", fontsize=label_font_size)
plt.ylabel("Model (config)", fontsize=label_font_size)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
errors_violin_path_all = os.path.join(comparison_dir, "Prediction_Errors_Violin_All_Configurations.png")
plt.savefig(errors_violin_path_all, dpi=300)
plt.close()
print(f"Saved prediction errors violin plot: {errors_violin_path_all}")

# Type 2: Violin plot for absolute prediction errors (Averaged by Model Type)
print("Generating Prediction Errors Violin Plot (Averaged by Model Type)...")
errors_df_avg = pd.concat([
    pd.DataFrame({
        "Error": data["Error"],
        "Model": data["Model"]
    }) for data in errors_data
])
plt.figure(figsize=(11.69, 8.27))
sns.violinplot(x="Error", y="Model", data=errors_df_avg, palette=custom_palette[:len(errors_df_avg["Model"].unique())])
plt.title("Absolute Prediction Errors by Model Type", fontsize=label_font_size)
plt.xlabel("Absolute Prediction Error (meters)", fontsize=label_font_size)
plt.ylabel("Model", fontsize=label_font_size)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
errors_violin_path_avg = os.path.join(comparison_dir, "Prediction_Errors_Violin_Averaged_Models.png")
plt.savefig(errors_violin_path_avg, dpi=300)
plt.close()
print(f"Saved prediction errors violin plot: {errors_violin_path_avg}")

# Step 3: Final Summary
print("\nStep 3: Generating Final Summary...")
final_summary = {
    "Total Rows Processed": len(combined_df),
    "Features Used": len(feature_cols),
    "Models Evaluated": len(results),
    "Total Plots Generated": len(plt.get_fignums()) + sum(len(results[(m, c)][14]) for m, c in results),
    "Execution Time (seconds)": round(time.time() - start_time, 2)
}
summary_df = pd.DataFrame([final_summary])
summary_path = os.path.join(modified_output_base_dir, "Execution_Summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Saved execution summary: {summary_path}")

# Save dataset statistics
data_stats_path = os.path.join(modified_output_base_dir, "Dataset_Statistics.csv")
data_stats_df.to_csv(data_stats_path, index=False)
print(f"Saved dataset statistics: {data_stats_path}")

print("Analysis complete.")