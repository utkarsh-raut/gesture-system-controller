import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib


X = []
y = []

# load dataset
with open("data/gestures.csv", "r") as f:
    reader = csv.reader(f)

    for row in reader:
        y.append(int(row[0]))  # label
        X.append([float(i) for i in row[1:]])

X = np.array(X)
y = np.array(y)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# model
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)

# train
model.fit(X_train, y_train)

# accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# save model
joblib.dump(model, "gesture_model.pkl")

print("Model saved as gesture_model.pkl")
