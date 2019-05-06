import mglearn

# Load City Bike datasets
citibike = mglearn.datasets.load_citibike()
print(citibike.head(20))

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10,3))

import pandas as pd
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')
week = ["SUN", "MON", "TUE", "WE", "THU", "FRI", "SAT"]
xticks_name = [week[int(w)] + d for w, d in zip(xticks.strftime("%w"), xticks.strftime(" %m-%d"))]

plt.xticks(xticks, xticks_name, rotation=90, ha="left")
plt.plot(citibike, linewidth=1)
plt.xlabel("Date")
plt.ylabel("# of Rentals")
plt.show()


# Target
y = citibike.values
print(y.shape)

print(citibike.index.shape)
print(citibike.index.astype("int64").values.shape)
print(citibike.index.astype("int64").values.reshape(-1, 1).shape)
X = citibike.index.astype("int64").values.reshape(-1, 1) // 10**9

print(X[0])


"""Defin eval/plot function
"""
n_train = 184

def eval_on_features(features, target, regressor):
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]

    regressor.fit(X_train, y_train)
    print("Test R^2: {:.2f}".format(regressor.score(X_test, y_test)))

    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    plt.figure(figsize=(10, 3))
    plt.xticks(range(0, len(X), 8), xticks_name, rotation=90, ha="left")

    plt.plot(range(n_train), y_train, label="Train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label='Test')

    plt.plot(range(n_train), y_pred_train, '--', label='Train Prediction')
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label='Test Prediction')

    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("# of Rentals")

    plt.show()


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
eval_on_features(X, y, regressor)


"""Use Expertise
"""

# Use hour
X_hour = citibike.index.hour.values.reshape(-1, 1)
eval_on_features(X_hour, y, regressor)

# Add day-of-week
import numpy as np
X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1, 1),
                         X_hour])
eval_on_features(X_hour_week, y, regressor)

