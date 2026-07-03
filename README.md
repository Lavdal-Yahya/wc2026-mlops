# WC 2026 Forecasting — MLOps Project

Prédiction de l'issue des matchs internationaux de football (victoire / nul / défaite)
en vue de la Coupe du Monde 2026. Pipeline MLOps complet : MLflow (tracking + registry
sur EC2), DVC + S3 (versioning des données), Flask (interface de prédiction).

> README en construction — sections infra/entraînement/déploiement à compléter par
> Lavdal, Mohamed et Mounaa. La section données ci-dessous est maintenue par Souleyman.

## Problème traité

*(TODO Mounaa — voir `docs/rapport_section_dataset.md` pour la version rapport)*

## Architecture technique

*(TODO Lavdal — schéma L9)*

## Données & prétraitement (Souleyman)

**Dataset** : [International football results 1872–2026](https://github.com/martj42/international_results)
— ≈ 49 500 matchs internationaux (date, équipes, score, compétition, terrain neutre).
Le script télécharge automatiquement le CSV brut s'il est absent.

**Deux versions de features** (pilotées par `preprocess.feature_version` dans `params.yaml`,
versionnées avec DVC) :

| Version | Features (avant-match uniquement — aucune fuite de données) |
|---|---|
| **v1** | Elo domicile/extérieur (K=20, bonus domicile 60 pts), différence d'Elo, terrain neutre |
| **v2** | v1 + forme sur 10 matchs (points, buts pour/contre) + head-to-head sur 5 confrontations (taux de victoire, diff. de buts) |

**Split temporel** (pas de mélange aléatoire) : train **1994 → 2022** (26 304 matchs),
test **2023 → 2026** (3 683 matchs), coupure au `2023-01-01`.

**Sorties** (`data/processed/`) :
- `train.csv` / `test.csv` — features + cible `outcome` (0 = victoire domicile, 1 = nul, 2 = victoire extérieure) ;
- `ratings_snapshot.joblib` — Elo / forme / h2h à jour par équipe, utilisé par Flask pour
  reconstruire le vecteur de features à partir de deux noms d'équipes ;
- `dataset_summary_<version>.json` — stats loggées aussi dans MLflow.

**Lancer le prétraitement** :

```bash
pip install -r requirements.txt
# facultatif : export MLFLOW_TRACKING_URI=http://<EC2>:5000  (sinon store local)
python scripts/preprocess.py                        # version de params.yaml (v1)
python scripts/preprocess.py --feature-version v2   # forcer la v2
```

Chaque exécution logge dans MLflow (expérience `wc2026-forecasting`, run `preprocess-v1|v2`) :
hyperparamètres Elo, fenêtres, dates de coupure, tailles train/test, répartition des classes.

## Entraînement & Model Registry

*(TODO Mohamed — `scripts/train.py`, sélection par log loss, registre `wc-outcome-model`)*

## Évaluation

*(TODO Mounaa — `scripts/evaluate.py`, calibration + matrice de confusion)*

## Déploiement Flask

*(TODO Lavdal — EC2 #2, formulaire domicile/extérieur/neutre → probabilités)*

## Reproduire l'environnement

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/preprocess.py
```

## Équipe

| Membre | Rôle | Branche |
|---|---|---|
| Lavdal | Infra AWS, DVC, Flask, intégration | `feat/infra-flask-lavdal` |
| Souleyman | Données & features | `feat/preprocess-souleyman` |
| Mohamed | Entraînement & registry | `feat/train-mohamed` |
| Mounaa | Évaluation & docs | `feat/evaluate-mounaa` |
