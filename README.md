# Test technique - Quantmetry

# Introduction

Vous trouverez dans ce README des éléments de code pour le test technique de Quantmetry (prédire le succès ou l'échec d'une candidature pour un poste de chercheur d'or chez "OrFée")

# Commandes

Création de l'environnement virtuel et installation des dépendances

```bash
pip install virtualenv
virtualenv .venv
source .venv/bin/activate # activation de l'environnement virtuel

pip install -r requirements.txt # installation des dépendances
```

Lancer le script d'entraînement, arguments:
* `--path_to_data`, str
* `--cv_splits`, int
* `--random_state`, int
* `--n_estimators`, int
* `--max_depth`, int  

```bash
python main.py 
```

Construction de l'image docker et déploiement du modèle sous forme d'api

```bash
docker image build . --tag quantmetry:0

docker container run --rm --name baseline_orfee -p 8080:8080 quantmetry:0

curl -X 'POST' \
  'http://localhost:8080/model/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 36,
  "exp": 5,
  "salaire": 23545,
  "diplome": "doctorat",
  "specialite": "forage",
  "note": 62,
  "dispo": "non"
}'

# {
#  "scores": [
#    0.542347186373107,
#    0.4576528136268929
#  ]
#}

```
