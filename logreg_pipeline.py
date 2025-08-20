# MNIST Logistic Regression — End-to-End Pipeline (Script Version)\n\n# Run: python mnist_logreg_pipeline.py\n\n# Adjust DATA_PATH and LABEL_COLUMN near the top if needed.\n\n
# ==== Step 0: Imports & Global Config ====
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import joblib

# Display options
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)

RANDOM_STATE = 42

# === Path settings ===
# Change this if your file is in a different location or has a different name.
# Examples:
#   'datasets/mnist.csv'
#   'datasets/mnist_train.csv'
#   'datasets/mnist_784.csv'
DATA_PATH = 'datasets/mnist.csv'   # <-- update if needed
LABEL_COLUMN = 'label'             # <-- change to your target column name if different
\n\n
# ==== Step 1: Load Data ====
# We load a CSV using pandas. Adjust DATA_PATH above if necessary.
assert os.path.exists(os.path.dirname(DATA_PATH)) or os.path.exists(DATA_PATH), (
    f"Path not found. Update DATA_PATH. Current: {DATA_PATH}"
)

# If the exact file doesn't exist, try to guess one from the datasets/ folder.
if not os.path.exists(DATA_PATH) and os.path.isdir(os.path.dirname(DATA_PATH)):
    candidates = [f for f in os.listdir(os.path.dirname(DATA_PATH)) if f.endswith('.csv')]
    if len(candidates) > 0:
        DATA_PATH = os.path.join(os.path.dirname(DATA_PATH), candidates[0])
        print(f"DATA_PATH not found; using first CSV found instead: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print('Loaded file:', DATA_PATH)
print('Shape:', df.shape)
display(df.head())
\n\n
# ==== Step 1.1: Confirm/Detect Label Column ====
# If LABEL_COLUMN doesn't exist, try to guess a label column.
if LABEL_COLUMN not in df.columns:
    common_names = ['label', 'target', 'digit', 'y', 'class']
    guessed = None
    for name in common_names:
        if name in df.columns:
            guessed = name
            break
    if guessed is None:
        # Heuristic: pick a column with small integer unique values (<=20 unique)
        int_like = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= 20]
        if int_like:
            guessed = int_like[0]

    if guessed is None:
        raise ValueError("Could not find a label column. Set LABEL_COLUMN to your target column name.")
    else:
        LABEL_COLUMN = guessed
        print(f"Auto-detected label column: {LABEL_COLUMN}")
else:
    print(f"Using label column: {LABEL_COLUMN}")
\n\n
# ==== Step 2: Basic EDA & Sanity Checks ====
print('\nData Types:')
print(df.dtypes.head(20))

print('\nNull counts (top 20 cols):')
print(df.isna().sum().head(20))

# Duplicates
dup_count = df.duplicated().sum()
print(f"\nDuplicate rows: {dup_count}")

# Label distribution
y_series = df[LABEL_COLUMN]
print('\nLabel distribution (counts):')
print(y_series.value_counts().sort_index())

print('\nLabel distribution (proportion):')
print((y_series.value_counts(normalize=True).sort_index() * 100).round(2).astype(str) + '%')

# Quick numeric description for the first 20 feature columns (excluding label)
feature_cols = [c for c in df.columns if c != LABEL_COLUMN]
print('\nDescribe (first 20 features):')
print(df[feature_cols[:20]].describe())

# Pixel range check (assuming pixel intensities 0-255 or 0-1)
min_val = df[feature_cols].min().min()
max_val = df[feature_cols].max().max()
print(f"\nFeature value range (overall): min={min_val}, max={max_val}")
\n\n# (Plot) Label Distribution\n\n
# ==== Step 2.1: Plot Label Distribution ====
# (Matplotlib only, one chart per figure, no manual colors)
label_counts = y_series.value_counts().sort_index()
plt.figure()
label_counts.plot(kind='bar')
plt.title('Label Distribution')
plt.xlabel('Digit')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
\n\n# (Plot) Sample Images if square like 28x28\n\n
# ==== Step 2.2: Visualize a Few Samples (if square image, e.g., 28x28) ====
n_features = len(feature_cols)
side = int(math.sqrt(n_features))

if side * side == n_features:  # square image
    # Show a few samples
    for idx in range(3):
        img = df.iloc[idx][feature_cols].values.reshape(side, side)
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title(f"Sample idx={idx}  Label={df.iloc[idx][LABEL_COLUMN]}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
else:
    print(f"Not visualizing samples: feature count {n_features} is not a perfect square.")
\n\n
# ==== Step 3: Train/Test Split (with stratification) ====
X = df.drop(columns=[LABEL_COLUMN])
y = df[LABEL_COLUMN].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print('Train shapes:', X_train.shape, y_train.shape)
print('Test shapes :', X_test.shape, y_test.shape)

print('\nClass proportions (train):')
print(y_train.value_counts(normalize=True).sort_index().round(3))

print('\nClass proportions (test):')
print(y_test.value_counts(normalize=True).sort_index().round(3))
\n\n
# ==== Step 4: Preprocessing + Logistic Regression Pipeline ====
# We standardize features before logistic regression.
logreg = LogisticRegression(
    penalty='l2',
    C=1.0,                 # inverse regularization strength
    solver='saga',         # good for multinomial + large datasets
    multi_class='multinomial',
    max_iter=1000,
    n_jobs=None,           # keep default; some versions don't parallelize here
    random_state=RANDOM_STATE
)

pipe = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),  # sparse-friendly; many MNIST CSVs are dense but this is robust
    ('logreg', logreg)
])

pipe
\n\n
# ==== Step 5: Train & Evaluate ====
pipe.fit(X_train, y_train)

y_pred_train = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print(f"Train accuracy: {acc_train:.4f}")
print(f"Test  accuracy: {acc_test:.4f}")

print('\nClassification Report (Test):')
print(classification_report(y_test, y_pred_test))
\n\n# (Plot) Confusion Matrix\n\n
# ==== Step 5.1: Confusion Matrix (matplotlib only) ====
cm = confusion_matrix(y_test, y_pred_test, labels=pipe.named_steps['logreg'].classes_)

plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix (Test)')
plt.colorbar()
tick_marks = np.arange(len(pipe.named_steps['logreg'].classes_))
plt.xticks(tick_marks, pipe.named_steps['logreg'].classes_, rotation=45)
plt.yticks(tick_marks, pipe.named_steps['logreg'].classes_)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()
\n\n
# ==== Step 6: Inspect Key Attributes/Methods ====
mdl = pipe.named_steps['logreg']

print('Classes_: ', mdl.classes_)
print('Intercept shape:', mdl.intercept_.shape)
print('Coef_ shape    :', mdl.coef_.shape)  # shape: (n_classes, n_features)

# Visualize weights as images per class if features form a square (e.g., 784 -> 28x28)
n_features = mdl.coef_.shape[1]
side = int(math.sqrt(n_features))

if side * side == n_features:
    for i, cls in enumerate(mdl.classes_):
        w = mdl.coef_[i].reshape(side, side)
        plt.figure()
        plt.imshow(w)
        plt.title(f'Weight map for class {cls}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
else:
    print(f"Skipping weight visualization; feature count {n_features} is not a perfect square.")
\n\n# Hyperparameter Tuning\n\n
# ==== Step 7: Hyperparameter Tuning with LogisticRegressionCV ====
# We'll wrap it in a pipeline to keep scaling consistent.
# Cs=10 tries a grid of C values by default; you can also pass a list, e.g., Cs=[0.01,0.1,1,10].
pipe_cv = make_pipeline(
    StandardScaler(with_mean=False),
    LogisticRegressionCV(
        Cs=10,
        cv=3,
        penalty='l2',
        solver='saga',
        multi_class='multinomial',
        max_iter=1000,
        random_state=RANDOM_STATE
    )
)

pipe_cv.fit(X_train, y_train)

cv_model = pipe_cv.named_steps['logisticregressioncv']
print('Chosen C per class (may vary):', cv_model.C_)

y_pred_cv = pipe_cv.predict(X_test)
print('\nAccuracy (LogisticRegressionCV on Test):', accuracy_score(y_test, y_pred_cv))
print('\nClassification Report (CV Model on Test):')
print(classification_report(y_test, y_pred_cv))
\n\n# (Plot) Learning Curve\n\n
# ==== Step 8: Learning Curve (under/overfitting diagnostic) ====
# We'll compute learning curves for the baseline pipeline.
train_sizes, train_scores, test_scores = learning_curve(
    pipe, X, y, cv=3, train_sizes=np.linspace(0.1, 1.0, 5), random_state=RANDOM_STATE
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, marker='o', label='Training score')
plt.plot(train_sizes, test_mean, marker='o', label='Cross-validation score')
plt.title('Learning Curve — Logistic Regression Pipeline')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.show()
\n\n# Save & Load\n\n
# ==== Step 9: Save & Load the Model ====
MODEL_PATH = 'mnist_logreg_pipeline.joblib'
joblib.dump(pipe, MODEL_PATH)
print(f"Saved pipeline to: {MODEL_PATH}")

# Example: load and use the model
loaded = joblib.load(MODEL_PATH)
print('Loaded model. Test accuracy again:', accuracy_score(y_test, loaded.predict(X_test)))
\n\n# Predict a few samples\n\n
# ==== Step 10: Predict a Few Samples ====
for idx in range(3):
    sample = X_test.iloc[[idx]]
    true_label = y_test.iloc[idx]
    pred_label = pipe.predict(sample)[0]
    proba = pipe.predict_proba(sample)[0]
    print(f"idx={idx}  true={true_label}  pred={pred_label}  top3 classes/probas:")
    # top-3 probabilities
    top3 = np.argsort(proba)[-3:][::-1]
    for j in top3:
        print(f"  class {mdl.classes_[j]}: {proba[j]:.4f}")
\n\n# --- End of Script ---