from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

LABEL_MAP={"edge_1": 0, "edge_2": 1, "cloud": 2}
REVERSE_LABEL_MAP = {0: "edge_1", 1: "edge_2", 2: "cloud"}

def train_model(dataset):
    X= dataset.drop(columns="target")
    y = dataset["target"].map(LABEL_MAP)
    # Split the sets for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss"
        )
    # Train the model
    model.fit(X_train, y_train)

    return model, X_test, y_test