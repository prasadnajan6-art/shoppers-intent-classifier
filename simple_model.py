# ================================================================
# ONLINE SHOPPERS PURCHASING INTENTION
# Simple Classification Model using Decision Tree
# ================================================================
# This script:
#   1. Downloads the dataset from the internet
#   2. Cleans and prepares the data
#   3. Trains a Decision Tree model
#   4. Shows how accurate our model is
#   5. Saves the model so the webpage can use it
#
# HOW TO RUN:
#   py simple_model.py
# ================================================================

# --- STEP 1: Import the tools we need ---
# Think of these as apps we're opening before we start working

import pandas as pd                          # For loading and working with data (like Excel)
import joblib                                # For saving our trained model to a file
from sklearn.tree import DecisionTreeClassifier      # Our ML model
from sklearn.model_selection import train_test_split # To split data into train & test
from sklearn.preprocessing import LabelEncoder       # To convert text to numbers
from sklearn.metrics import accuracy_score, classification_report  # To measure results

print("=" * 55)
print("  ONLINE SHOPPERS PURCHASING INTENTION")
print("  Simple Decision Tree Classifier")
print("=" * 55)

# --- STEP 2: Load the Dataset ---
# We are downloading the UCI dataset directly from the internet
# It has 12,330 rows — each row is one website visit session

print("\n📦 Step 1: Loading the dataset...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
data = pd.read_csv(url)

print(f"   ✅ Dataset loaded! {data.shape[0]} rows, {data.shape[1]} columns")
print(f"   📊 Purchase rate: {data['Revenue'].mean()*100:.1f}% of visits result in a purchase")

# --- STEP 3: Prepare the Data ---
# Computers can only work with numbers, not words
# So we convert text columns (like "Month", "VisitorType") into numbers

print("\n🔧 Step 2: Preparing the data...")

# Convert text to numbers using LabelEncoder
# Example: "Jan"=0, "Feb"=1, "Mar"=2 ... etc
le = LabelEncoder()
data['Month']       = le.fit_transform(data['Month'])
data['VisitorType'] = le.fit_transform(data['VisitorType'])

# Convert True/False to 1/0
data['Weekend'] = data['Weekend'].astype(int)
data['Revenue'] = data['Revenue'].astype(int)  # This is our TARGET (what we predict)

print("   ✅ Text columns converted to numbers")

# --- STEP 4: Split into Features and Target ---
# Features (X) = the inputs  → everything we know about a visit
# Target   (y) = the output  → did they buy? (1=Yes, 0=No)

X = data.drop('Revenue', axis=1)   # All columns EXCEPT Revenue
y = data['Revenue']                 # Only the Revenue column

print(f"   📌 Features (inputs): {X.shape[1]} columns")
print(f"   🎯 Target (output): Revenue (1=Purchase, 0=No Purchase)")

# --- STEP 5: Split into Training and Testing sets ---
# We train on 80% of the data, and test on the remaining 20%
# This way we can check if our model works on data it has never seen

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # Makes results consistent every time
    stratify=y          # Keep same ratio of buyers/non-buyers in both sets
)

print(f"\n📂 Step 3: Splitting data...")
print(f"   Training set : {len(X_train)} rows (80%)")
print(f"   Testing set  : {len(X_test)} rows (20%)")

# --- STEP 6: Train the Model ---
# A Decision Tree is like a flowchart of YES/NO questions
# Example: "Is Page Value > 10?" → Yes → "Is it November?" → Yes → Predict: Purchase
# The model learns these questions automatically from the training data

print(f"\n🤖 Step 4: Training the Decision Tree model...")

model = DecisionTreeClassifier(
    max_depth=5,        # Limit tree depth so it doesn't overfit
    random_state=42     # Makes results consistent
)
model.fit(X_train, y_train)  # This is where the actual "learning" happens

print("   ✅ Model trained successfully!")

# --- STEP 7: Make Predictions and Evaluate ---
# Now we test our model on the 20% of data it has never seen

print(f"\n📊 Step 5: Evaluating the model...")

predictions = model.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)

print(f"\n   ✅ Accuracy: {accuracy*100:.2f}%")
print(f"   This means the model correctly predicted {accuracy*100:.1f}% of sessions!\n")
print("   Detailed Report:")
print(classification_report(y_test, predictions, target_names=["No Purchase", "Purchase"]))

# --- STEP 8: Show Top Features ---
# Which columns were most useful for the model?

print("🔍 Most Important Features (what the model looks at most):")
importances = pd.Series(model.feature_importances_, index=X.columns)
top5 = importances.sort_values(ascending=False).head(5)
for i, (feature, score) in enumerate(top5.items(), 1):
    print(f"   {i}. {feature:30s} → {score*100:.1f}%")

# --- STEP 9: Save the Model ---
# We save the model to a file so the webpage can load it and make predictions

joblib.dump(model, 'simple_model.pkl')
print(f"\n💾 Model saved as 'simple_model.pkl'")
print(f"\n✅ All done! Now run: py simple_api.py")
print("=" * 55)
