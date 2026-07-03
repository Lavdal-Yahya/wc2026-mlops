# Présentation du problème ML et du dataset

*(Section du mini-rapport — rédigée par Souleyman, tâche S3)*

## Le problème

Nous voulons **prédire l'issue d'un match international de football** — victoire de
l'équipe à domicile, match nul, ou victoire à l'extérieur — dans la perspective de la
**Coupe du Monde 2026**. C'est un problème de **classification multiclasse à 3 classes**,
où le modèle produit une probabilité pour chaque issue : P(victoire), P(nul), P(défaite).
Ces probabilités alimentent une application web (Flask) où l'utilisateur choisit deux
équipes et un lieu (terrain neutre ou non), ainsi qu'une éventuelle simulation du tournoi.

## Le dataset

Nous utilisons le jeu de données public **« International football results 1872–2026 »**
(martj42, GitHub) : **≈ 49 500 matchs officiels** entre équipes nationales, avec pour
chaque match la date, les deux équipes, le score, la compétition et un indicateur de
terrain neutre. Les matchs futurs sans score (calendrier CM 2026) sont écartés.

Le score brut n'est pas directement exploitable comme entrée : nous construisons des
**features d'avant-match**, calculées uniquement à partir des matchs *antérieurs* pour
éviter toute fuite de données (data leakage) :

- **v1 (Elo uniquement)** : classement Elo de chaque équipe mis à jour match après match
  (K=20, bonus domicile de 60 points hors terrain neutre), différence d'Elo, terrain neutre.
- **v2 (Elo + forme + confrontations)** : ajoute la forme récente (points, buts marqués et
  encaissés sur les 10 derniers matchs) et l'historique des confrontations directes
  (taux de victoire et différence de buts sur les 5 dernières rencontres).

Les deux versions sont produites par le même script (`scripts/preprocess.py`), pilotées
par un simple drapeau dans `params.yaml`, et **versionnées avec DVC** (data v1 / data v2).

Le découpage train/test est **temporel** et non aléatoire : entraînement sur
**1994 → 2022** (26 304 matchs), test sur **2023 → 2026** (3 683 matchs). Un découpage
aléatoire serait invalide ici : il permettrait au modèle de « voir le futur » (des matchs
de 2024 en entraînement pour prédire 2019), alors qu'en production on prédit toujours
des matchs à venir. Le prétraitement produit aussi un **instantané des ratings**
(`ratings_snapshot.joblib` : Elo, forme et confrontations à jour par équipe) qui permet à
l'application Flask de reconstruire le vecteur de features à partir de deux noms d'équipes.

## Pourquoi la log loss plutôt que l'accuracy ?

Les classes sont **déséquilibrées** (48,7 % victoires à domicile, 23,4 % nuls, 27,9 %
victoires extérieures) : un modèle naïf qui prédit toujours « victoire à domicile »
atteint déjà ≈ 49 % d'accuracy sans rien apprendre. Surtout, notre produit final est une
**probabilité**, pas une étiquette : annoncer « France 78 % » doit être fiable. La
**log loss** pénalise les probabilités mal calibrées et récompense un modèle qui doute à
bon escient, là où l'accuracy ignore complètement la confiance du modèle. Nous suivons
donc la log loss comme métrique principale de sélection dans MLflow (le meilleur modèle
enregistré au Model Registry est celui qui la minimise), complétée par le score de Brier,
l'accuracy et le F1 macro à titre indicatif.
