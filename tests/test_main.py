import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.inference import predict

def test_predict():
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1]
    })
    y = pd.Series([0, 1, 0, 1, 0])

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    predictions = predict(model, X)

    assert len(predictions) == len(X)
    assert all(pred in [0, 1] for pred in predictions)
    assert predictions.dtype == 'int64'
