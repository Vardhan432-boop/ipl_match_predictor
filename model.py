import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("matches.csv")

# Keep only needed columns
features = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'season']
target = 'winner'

df = df[features + [target]].dropna()

# If season looks like 2008/2009, extract first year
df['season'] = df['season'].astype(str).str.split('/').str[0].astype(int)

X = df[features]
y = df[target]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']),
        ('num', 'passthrough', ['season'])
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])

# Train
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model and label encoder
joblib.dump(model, "ipl_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Saved model successfully!")