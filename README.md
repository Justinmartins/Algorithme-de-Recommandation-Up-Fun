# 🧠 Système de Recommandation d’Activités

Ce projet présente la construction d’un système de recommandation d’activités personnalisées basé sur des données simulées et des modèles d’apprentissage supervisé.

## 📁 Structure du dépôt

- `Algo_Recommandation.ipynb` : Notebook principal contenant le pipeline d’entraînement, d’évaluation et de génération des recommandations.
- `activites.csv` : Liste d’activités classées par catégorie, avec likes et rating.
- `utilisateurs.csv` : 10 000 utilisateurs simulés rattachés à des personas types.
- `historique_utilisateur.csv` : Historique partiel des activités passées.
- `interactions_utilisateur.csv` : Notes explicites et dates d'interactions.
- `vendors.csv` : Données sur les prestataires d’activités.
- `Algorithme_de_recommandation_Documentation_Technique.pdf` : Documentation complète du projet.

## 👤 Personas simulés

Cinq profils types ont été créés pour simuler des comportements variés :
- **Léo** (29 ans, Lyon) : sportif (crossfit, escalade…)
- **Clara** (35 ans, Paris) : culturelle (musées, expos…)
- **Jeanne** (42 ans, Nantes) : manuelle (poterie, couture…)
- **Mehdi** (38 ans, Montpellier) : bien-être (yoga, spa…)
- **Lucas** (27 ans, Marseille) : extrême (escape game, simulateur…)

## ⚙️ Méthodologie

1. **Simulation des données** (activités, utilisateurs, interactions)
2. **Prétraitement** : encodage des catégories, gestion des valeurs manquantes
3. **Entraînement des modèles** :
   - Random Forest Regressor
   - LightGBM (meilleur RMSE)
   - XGBoost
   - Régression linéaire (baseline)
4. **Évaluation** : RMSE (Root Mean Squared Error)
5. **Génération des recommandations** : top 5 activités prédites pour chaque utilisateur

## 📊 Résultats

| Modèle              | RMSE    |
|---------------------|---------|
| LightGBM            | **0.4815** |
| XGBoost             | 0.4966  |
| Régression Linéaire | 0.4932  |
| Random Forest       | 0.5646  |

## 🔍 Pistes d'amélioration

- Intégration de données temporelles (référence, saison)
- Ajout de variables utilisateurs historiques (fréquence, moyennes)
- Modèles collaboratifs avancés (Surprise SVD, LightFM)
