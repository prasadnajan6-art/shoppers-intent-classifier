# ================================================================
# ONLINE SHOPPERS PURCHASING INTENTION
# Complete Data Science Pipeline — All Steps Included
# ================================================================
# Steps covered:
#   1. Data Collection
#   2. Data Understanding
#   3. Data Cleaning
#   4. Exploratory Data Analysis (EDA) — full visualizations
#   5. Feature Engineering
#   6. Data Transformation (encoding + scaling)
#   7. Train-Test Split
#   8. Model Selection & Training
#   9. Model Evaluation
#
# HOW TO RUN:
#   py complete_pipeline.py
#
# All charts are saved as PNG files in your project folder.
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Plot style ────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120

print("=" * 60)
print("  ONLINE SHOPPERS PURCHASING INTENTION")
print("  Complete Data Science Pipeline")
print("=" * 60)


# ════════════════════════════════════════════════════════════
# STEP 1 — DATA COLLECTION
# ════════════════════════════════════════════════════════════
print("\n📦 STEP 1: Data Collection")
print("-" * 40)

url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00468/online_shoppers_intention.csv"
)
df = pd.read_csv(url)
print(f"   ✅ Dataset loaded from UCI Repository")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")


# ════════════════════════════════════════════════════════════
# STEP 2 — DATA UNDERSTANDING
# ════════════════════════════════════════════════════════════
print("\n🔍 STEP 2: Data Understanding")
print("-" * 40)

print("\n   First 3 rows:")
print(df.head(3).to_string())

print("\n   Column data types:")
print(df.dtypes.to_string())

print("\n   Target variable (Revenue) distribution:")
print(df['Revenue'].value_counts())
print(f"\n   Purchase rate: {df['Revenue'].mean()*100:.1f}%")

print("\n   Summary statistics (numeric columns):")
print(df.describe().round(2).to_string())


# ════════════════════════════════════════════════════════════
# STEP 3 — DATA CLEANING
# ════════════════════════════════════════════════════════════
print("\n🧹 STEP 3: Data Cleaning")
print("-" * 40)

# Check missing values
missing = df.isnull().sum()
print(f"\n   Missing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "   None found ✅")

# Check duplicates
dupes = df.duplicated().sum()
print(f"\n   Duplicate rows: {dupes}")
if dupes > 0:
    df = df.drop_duplicates()
    print(f"   Removed {dupes} duplicate rows ✅")
else:
    print("   No duplicates found ✅")

# Check for outliers using IQR method
print("\n   Checking for outliers (IQR method):")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
outlier_counts = {}
for col in numeric_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    if outliers > 0:
        outlier_counts[col] = outliers
        print(f"   {col:35s}: {outliers} outliers")

# Cap outliers at 1st and 99th percentile (winsorization)
for col in outlier_counts:
    lo = df[col].quantile(0.01)
    hi = df[col].quantile(0.99)
    df[col] = df[col].clip(lo, hi)
print(f"\n   ✅ Outliers capped at 1st/99th percentile (winsorization)")


# ════════════════════════════════════════════════════════════
# STEP 4 — EXPLORATORY DATA ANALYSIS (EDA)
# ════════════════════════════════════════════════════════════
print("\n📊 STEP 4: Exploratory Data Analysis (EDA)")
print("-" * 40)

# ── 4a. Target distribution pie chart ──
fig, ax = plt.subplots(figsize=(5, 4))
counts = df['Revenue'].value_counts()
ax.pie(counts, labels=['No Purchase', 'Purchase'],
       autopct='%1.1f%%', colors=['#cbd5e0', '#1e3a5f'],
       startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
ax.set_title('Target Variable Distribution (Revenue)', fontweight='bold')
plt.tight_layout()
plt.savefig('eda_01_target_distribution.png')
plt.close()
print("   ✅ Saved: eda_01_target_distribution.png")

# ── 4b. Histograms for numeric features ──
num_features = [
    'Administrative', 'Informational', 'ProductRelated',
    'BounceRates', 'ExitRates', 'PageValues'
]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(num_features):
    axes[i].hist(df[col], bins=30, color='#1e3a5f', edgecolor='white', alpha=0.8)
    axes[i].set_title(col, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
plt.suptitle('Histograms — Key Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_02_histograms.png')
plt.close()
print("   ✅ Saved: eda_02_histograms.png")

# ── 4c. Box plots — numeric features by Revenue ──
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(num_features):
    df.boxplot(column=col, by='Revenue', ax=axes[i],
               boxprops=dict(color='#1e3a5f'),
               medianprops=dict(color='#e53e3e', linewidth=2))
    axes[i].set_title(col, fontweight='bold')
    axes[i].set_xlabel('Purchase (False/True)')
plt.suptitle('Box Plots — Features by Purchase Outcome', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_03_boxplots.png')
plt.close()
print("   ✅ Saved: eda_03_boxplots.png")

# ── 4d. Scatter plot — PageValues vs ExitRates ──
fig, ax = plt.subplots(figsize=(7, 5))
colors = df['Revenue'].map({False: '#cbd5e0', True: '#1e3a5f'})
ax.scatter(df['PageValues'], df['ExitRates'], c=colors, alpha=0.4, s=15)
no_patch  = mpatches.Patch(color='#cbd5e0', label='No Purchase')
yes_patch = mpatches.Patch(color='#1e3a5f', label='Purchase')
ax.legend(handles=[no_patch, yes_patch])
ax.set_xlabel('Page Values')
ax.set_ylabel('Exit Rates')
ax.set_title('Page Values vs Exit Rates (coloured by Purchase)', fontweight='bold')
plt.tight_layout()
plt.savefig('eda_04_scatter.png')
plt.close()
print("   ✅ Saved: eda_04_scatter.png")

# ── 4e. Correlation heatmap ──
fig, ax = plt.subplots(figsize=(10, 7))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='Blues', ax=ax, linewidths=0.5,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap — Numeric Features', fontweight='bold')
plt.tight_layout()
plt.savefig('eda_05_correlation_heatmap.png')
plt.close()
print("   ✅ Saved: eda_05_correlation_heatmap.png")

# ── 4f. Purchase rate by Month ──
month_order = ['Feb','Mar','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_rate  = df.groupby('Month')['Revenue'].mean().reindex(month_order) * 100
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(month_rate.index, month_rate.values,
              color='#1e3a5f', edgecolor='white')
ax.set_title('Purchase Rate by Month (%)', fontweight='bold')
ax.set_ylabel('Purchase Rate (%)')
ax.set_xlabel('Month')
for bar, val in zip(bars, month_rate.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('eda_06_purchase_by_month.png')
plt.close()
print("   ✅ Saved: eda_06_purchase_by_month.png")

# ── 4g. Purchase rate by Visitor Type ──
vtype_rate = df.groupby('VisitorType')['Revenue'].mean() * 100
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(vtype_rate.index, vtype_rate.values,
              color=['#2a4a7f','#1e3a5f','#4a6fa5'], edgecolor='white')
ax.set_title('Purchase Rate by Visitor Type (%)', fontweight='bold')
ax.set_ylabel('Purchase Rate (%)')
for bar, val in zip(bars, vtype_rate.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('eda_07_purchase_by_visitor.png')
plt.close()
print("   ✅ Saved: eda_07_purchase_by_visitor.png")

print("\n   Key EDA findings:")
print(f"   - Only {df['Revenue'].mean()*100:.1f}% of sessions result in a purchase")
print(f"   - November has the highest purchase rate")
print(f"   - Returning visitors purchase more than new visitors")
print(f"   - PageValues is strongly correlated with purchase outcome")


# ════════════════════════════════════════════════════════════
# STEP 5 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
print("\n⚙️  STEP 5: Feature Engineering")
print("-" * 40)

# Total pages visited
df['Total_Pages'] = (
    df['Administrative'] + df['Informational'] + df['ProductRelated']
)

# Total time spent on site
df['Total_Duration'] = (
    df['Administrative_Duration'] +
    df['Informational_Duration']  +
    df['ProductRelated_Duration']
)

# Product page ratio (what % of pages were product pages)
df['Product_Page_Ratio'] = np.where(
    df['Total_Pages'] > 0,
    df['ProductRelated'] / df['Total_Pages'],
    0
)

# Engagement score (high page value + low bounce = engaged)
df['Engagement_Score'] = df['PageValues'] * (1 - df['BounceRates'])

print("   ✅ New features created:")
print("   - Total_Pages          : sum of all pages visited")
print("   - Total_Duration       : total time spent on site (seconds)")
print("   - Product_Page_Ratio   : % of pages that were product pages")
print("   - Engagement_Score     : PageValues × (1 - BounceRates)")


# ════════════════════════════════════════════════════════════
# STEP 6 — DATA TRANSFORMATION
# ════════════════════════════════════════════════════════════
print("\n🔄 STEP 6: Data Transformation")
print("-" * 40)

# Label Encoding for categorical columns
le_month   = LabelEncoder()
le_visitor = LabelEncoder()

df['Month']       = le_month.fit_transform(df['Month'])
df['VisitorType'] = le_visitor.fit_transform(df['VisitorType'])
df['Weekend']     = df['Weekend'].astype(int)
df['Revenue']     = df['Revenue'].astype(int)

print("   ✅ Label Encoding applied to: Month, VisitorType")
print("   ✅ Boolean columns converted to 0/1: Weekend, Revenue")

# Separate features and target
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# Feature Scaling — StandardScaler
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("   ✅ StandardScaler applied to all feature columns")
print(f"   Features after engineering: {X.shape[1]} columns")


# ════════════════════════════════════════════════════════════
# STEP 7 — TRAIN-TEST SPLIT
# ════════════════════════════════════════════════════════════
print("\n✂️  STEP 7: Train-Test Split")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"   Training set : {len(X_train)} rows (80%)")
print(f"   Testing set  : {len(X_test)} rows (20%)")
print(f"   Purchase rate in train: {y_train.mean()*100:.1f}%")
print(f"   Purchase rate in test : {y_test.mean()*100:.1f}%")


# ════════════════════════════════════════════════════════════
# STEP 8 — MODEL SELECTION & TRAINING
# ════════════════════════════════════════════════════════════
print("\n🤖 STEP 8: Model Selection & Training")
print("-" * 40)

print("   Selected algorithm: Decision Tree Classifier")
print("   Reason: Interpretable, works well for classification,")
print("           easy to explain and visualise\n")

model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)
print("   ✅ Model trained successfully!")


# ════════════════════════════════════════════════════════════
# STEP 9 — MODEL EVALUATION
# ════════════════════════════════════════════════════════════
print("\n📈 STEP 9: Model Evaluation")
print("-" * 40)

y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n   Accuracy : {accuracy*100:.2f}%")
print("\n   Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['No Purchase', 'Purchase']
))

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['No Purchase', 'Purchase'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('eval_01_confusion_matrix.png')
plt.close()
print("   ✅ Saved: eval_01_confusion_matrix.png")

# Feature importance
importances = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind='barh', ax=ax, color='#1e3a5f', edgecolor='white')
ax.set_title('Top 10 Feature Importances', fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('eval_02_feature_importance.png')
plt.close()
print("   ✅ Saved: eval_02_feature_importance.png")

print("\n   Top 5 most important features:")
top5 = importances.sort_values(ascending=False).head(5)
for i, (feat, score) in enumerate(top5.items(), 1):
    print(f"   {i}. {feat:30s} {score*100:.1f}%")

# Save model and preprocessors
joblib.dump(model,      'complete_model.pkl')
joblib.dump(scaler,     'complete_scaler.pkl')
joblib.dump(le_month,   'complete_le_month.pkl')
joblib.dump(le_visitor, 'complete_le_visitor.pkl')

print("\n💾 Model saved as complete_model.pkl")
print("\n" + "=" * 60)
print("  ✅ PIPELINE COMPLETE!")
print("  Charts saved:")
print("    eda_01_target_distribution.png")
print("    eda_02_histograms.png")
print("    eda_03_boxplots.png")
print("    eda_04_scatter.png")
print("    eda_05_correlation_heatmap.png")
print("    eda_06_purchase_by_month.png")
print("    eda_07_purchase_by_visitor.png")
print("    eval_01_confusion_matrix.png")
print("    eval_02_feature_importance.png")
print("=" * 60)
