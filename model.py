import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score,mean_squared_error

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    
    metrics = {
        "coef_correlation": r2,
        "mean_square_error": mse,

    }

    return json.loads(
        json.dumps(metrics)
    )
