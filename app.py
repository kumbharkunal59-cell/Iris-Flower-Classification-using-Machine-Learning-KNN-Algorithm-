"""
Iris Flower Classification - Terminal Based Machine Learning Project

Author: Your Name
Description:
A simple terminal-based machine learning application that trains a KNN model
on the Iris dataset and allows users to predict flower species by entering
measurements.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import sys


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    print(Colors.BOLD)
    print("==========================================")
    print("        IRIS FLOWER CLASSIFIER")
    print("==========================================")
    print(Colors.END)


def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, iris, accuracy


def get_user_input():
    try:
        print("\nEnter flower measurements (in cm):")

        sepal_length = float(input("Sepal length: "))
        sepal_width = float(input("Sepal width: "))
        petal_length = float(input("Petal length: "))
        petal_width = float(input("Petal width: "))

        return [sepal_length, sepal_width, petal_length, petal_width]

    except ValueError:
        print("\nInvalid input. Please enter numeric values only.\n")
        return None


def main():
    clear_screen()
    print_header()

    print("Training model...")
    model, iris, accuracy = train_model()

    print("\nModel trained successfully.")
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    while True:
        user_data = get_user_input()

        if user_data:
            prediction = model.predict([user_data])
            flower_name = iris.target_names[prediction[0]]

            print("\nPrediction result:")
            print(f"Predicted flower species: {flower_name}\n")

        choice = input("Do you want to predict another flower? (y/n): ").lower()

        if choice != 'y':
            print("\nExiting application.")
            sys.exit()


if __name__ == "__main__":
    main()
