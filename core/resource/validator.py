
import pandas as pd
from sklearn.base import BaseEstimator
class ResourceValidator:

    @staticmethod
    def dataset(value) -> None:
        if not isinstance(value, pd.DataFrame):
            raise TypeError(
                "Expected pandas.DataFrame"
            )

    @staticmethod
    def model(value) -> None:
        if not isinstance(value, BaseEstimator):
            raise TypeError(
                "Expected scikit-learn estimator"
            )

        if not hasattr(value, "predict"):
            raise TypeError(
                "Expected estimator with predict()"
            )