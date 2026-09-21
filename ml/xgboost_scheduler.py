import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from ml.dataset import get_state_features

LABEL_MAP={"edge_1": 0, "edge_2": 1, "cloud": 2}
REVERSE_LABEL_MAP = {0: "edge_1", 1: "edge_2", 2: "cloud"}

def train_model(dataset):
    X= dataset.drop(columns="target")
    y = dataset["target"].map(LABEL_MAP)
    # Split the sets for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="mlogloss"
        )
    # Train the model
    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy:{accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(
            y_test, y_pred,
            target_names=["edge_1", "edge_2", "cloud"],
        )
    )

def predict_node(model, nodes, request):
    """
    Predict which node should handle the request
    """
    features = get_state_features(nodes, request)
    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    return nodes[int(prediction)]
