from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def train_churn_model(X_train, y_train):
    
    numeric_features = [
        "monthly_fee", "usage_hours", "support_requests",
        "account_age_months", "failed_payments", "autopay_enabled"
    ]
    
    categorical_features = ["region", "device_type", "payment_method"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline

    
def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "accuracy": float(accuracy),
        "f1_score": float(f1)
    }