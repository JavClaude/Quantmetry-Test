import json
import pickle
import logging
import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import precision_score
from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import Pipeline
from mlbox.preprocessing import Categorical_encoder
from sklearn.ensemble import RandomForestClassifier

from Utils.utils import QuantMetryPreprocesseur, chi_squared_test

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def main(**kwargs) -> None:

    np.random.seed(kwargs["random_state"])

    df = pd.read_csv(kwargs.pop("path_to_data"))

    # Suppression des variables inutiles / pas très éthiques 
    df.drop(
        columns=["Unnamed: 0", "index", "date","cheveux", "sexe"], 
        inplace=True  
    )

    df.dropna(inplace=True)

    # A) max_note: 100, note maximum possible
    # B) min_age: 16, pas légal de travailler avant de 16 ans
    # C) max_age: 67, "pas légal" de travailler après 67 ans
    # D) min_exp: pas possible d'avoir une expérience négatif
    # E) codif_diplome:
    #   *) doctorat: 8 ans
    #   *) master: 5 ans
    #   *) licence: 3 ans
    #   *) bac: 0 ans
    # F) diplome_heuristique: 17 & exp_heuristique: 16
    #   *) Des candidats ont un âge faible, un haut diplôme ainsi qu'un bon nombre d'année d'expérience

    preprocesseur_args = {
        "max_note": 100,
        "min_age": 16,
        "max_age": 67,
        "min_exp": 0,
        "codif_dip":  {
            "doctorat": 8,
            "master": 5,
            "licence": 3,
            "bac": 0
        },
        "diplome_heuristique": 17,
        "exp_heuristique": 16
    }

    QM = QuantMetryPreprocesseur(**preprocesseur_args)
    df_ = QM.transform(df)

    RandomForest = RandomForestClassifier(
        n_estimators=kwargs["n_estimators"],
        max_depth=kwargs["max_depth"],
        class_weight="balanced_subsample",
        random_state=kwargs["random_state"]
    )

    Model = Pipeline(
        [
            ("CatEncoder", Categorical_encoder("label_encoding")),
            ("RandomForest", RandomForest)
        ]
    )

    Cv = StratifiedKFold(n_splits=kwargs["cv_splits"], random_state=kwargs["random_state"], shuffle=True)

    X = df_.drop(columns="embauche")
    y = df_["embauche"]

    precisions_ = []

    for train_index, test_index in Cv.split(X, y):
        Model.fit(
                X.iloc[train_index],
            y.iloc[train_index]
        )

        y_pred = Model.predict_proba(
                X.iloc[test_index]
        )[:, 1]

        precisions_.append(
            precision_score(
                y.iloc[test_index], 
                np.where(
                    y_pred>0.65,
                    1,
                    0
                )
            )
        )

    logger.warning("Précision moyenne sur {} folds: {}".format(kwargs["cv_splits"], sum(precisions_)/len(precisions_)))    

    feature_importances = pd.DataFrame(
        data=[Model[1].feature_importances_],
        columns=X.columns
    )

    output_file = {
        **kwargs,
        "precisions": precisions_,
        "feature importances": {
            **feature_importances.to_dict()
        }
    }

    with open("output_file.json", "w") as file:
        json.dump(output_file, file)

    with open("Model.pkl", "wb") as file:
        pickle.dump(Model, file)

if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--path_to_data", type=str, required=False, default="Data/data_v1.0 (3).csv", help="Path to raw data")
    argument_parser.add_argument("--cv_splits", type=int, required=False, default=4)
    argument_parser.add_argument("--random_state", type=int, required=False, default=42)
    argument_parser.add_argument("--n_estimators", type=int, required=False, default=100)
    argument_parser.add_argument("--max_depth", type=int, required=False, default=5)

    arguments = argument_parser.parse_args()

    main(**vars(arguments))