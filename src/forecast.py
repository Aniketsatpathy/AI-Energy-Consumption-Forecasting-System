def forecast(model, X_test, steps=24):
    """
    Forecast future values
    """
    predictions = model.predict(X_test.tail(steps))
    return predictions