import logging

import pandas as pd
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class QuantMetryPreprocesseur(object):
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)
    
    def __repr__(self) -> str:
        return "QuantMetryPreprocessor"
    
    def __str__(self) -> str:
        return "QuantMetryPeprocessor"
    
    def _drop_fake_note(self, DataFrame: pd.DataFrame) -> pd.DataFrame:
        return DataFrame[
            DataFrame["note"]<self.max_note
        ]
    
    def _drop_non_legal_age(self, DataFrame: pd.DataFrame) -> pd.DataFrame:
        return DataFrame[
            (DataFrame["age"]>self.min_age) & (DataFrame["age"]<self.max_age)
        ]
    
    def _drop_fake_diplome(self, DataFrame: pd.DataFrame) -> pd.DataFrame:
        return DataFrame[
            (DataFrame["age"] - (DataFrame["diplome"].apply(lambda x: self.codif_dip[x]))>self.diplome_heuristique)
        ]
    
    def _drop_fake_exp(self, DataFrame: pd.DataFrame) -> pd.DataFrame:
        return DataFrame[
            ((DataFrame["age"]-DataFrame["exp"])>self.exp_heuristique) & (DataFrame["exp"] > self.min_exp)
        ]
    
    def transform(self, DataFrame: pd.DataFrame) -> pd.DataFrame:
        DataFrame = self._drop_fake_note(DataFrame)
        DataFrame = self._drop_non_legal_age(DataFrame)
        DataFrame = self._drop_fake_diplome(DataFrame)
        DataFrame = self._drop_fake_exp(DataFrame)
        return DataFrame

def chi_squared_test(X_1: pd.Series, X_2: pd.Series, prob: float) -> None:
    contingency_table = pd.crosstab(X_1, X_2)

    _, p, _, _ = chi2_contingency(contingency_table)

    if p <= (1-prob):
        logger.warning(
            "{} and {} are Dependent (H0 is rejected), p-value: {}".format(
                X_1.name, X_2.name, p
            )
        )

    else:
        logger.warning(
            "{} and {} are Independent (H0 is not rejected), p-value: {}".format(
                X_1.name, X_2.name, p
            )
        )