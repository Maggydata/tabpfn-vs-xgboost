"""Chargement et découpage des données, partagé par les notebooks et le pipeline."""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
N_SAMPLE = 2000
TEST_SIZE = 0.2
N_SHAP = 100
N_BACKGROUND = 20  # réduit de 50 à 20 : SHAP TabPFN trop lent sur M1


def load_split():
    df = fetch_california_housing(as_frame=True).frame
    df = df.sample(n=N_SAMPLE, random_state=RANDOM_STATE)

    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_shap = X_test.sample(n=N_SHAP, random_state=RANDOM_STATE)
    X_background = X_train.sample(n=N_BACKGROUND, random_state=RANDOM_STATE)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_shap": X_shap, "X_background": X_background,
    }
