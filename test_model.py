import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import os

#charge les fichiers csv
def load_and_prepare_data(folder_path="."):  # racine
    df_users = pd.read_csv(os.path.join(folder_path, "utilisateurs.csv")) 
    df_activities = pd.read_csv(os.path.join(folder_path, "activites.csv")) 
    df_interactions = pd.read_csv(os.path.join(folder_path, "interactions_utilisateur.csv")) 

    #fusion des 3 csv pour avoir un seul data frame df, permet de  faire des prédictions personnalisées, lier les infos
    df = df_interactions.merge(df_users, on="user_id", how="left")
    df = df.merge(df_activities, on="activity_id", how="left")

    label_encoders = {}
    for col in ["persona_category", "category", "location"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, df_users, df_activities, label_encoders

# Entraînement des modèles
def train_models(df):
    features = ["user_id", "activity_id", "category", "vendor_id", "likes", "rating"]
    target = "user_rating"

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "LightGBM": LGBMRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, verbosity=0),
        "Linear Regression": LinearRegression()
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = rmse
        trained_models[name] = model

    return results, trained_models

# Tests -------------------------

def test_data_loading():
    df, df_users, df_activities, le = load_and_prepare_data(".")
    assert not df.empty
    assert "user_id" in df.columns
    assert "activity_id" in df.columns
    assert "persona_category" in df.columns

def test_model_training():
    df, *_ = load_and_prepare_data(".") #chargement des données
    results, models = train_models(df)
    assert all(isinstance(rmse, float) for rmse in results.values()) #rmse = float
    assert all(hasattr(m, "predict") for m in models.values()) #modeles possede une methode predict = entrainé
    assert min(results.values()) > 0  # RMSE raisonnable : le plus petit RMSE > 0 

def test_recommendation_top5():
    df, df_users, df_activities, le = load_and_prepare_data(".") #chargement des données
    _, models = train_models(df) #entrainement des modele sur df 
    
    # Sélection du meilleur modèle selon le + petit RMSE
    best_model = models[min(models, key=lambda k: np.sqrt(
        mean_squared_error(df["user_rating"], models[k].predict(df[["user_id", "activity_id", "category", "vendor_id", "likes", "rating"]]))
    ))]

    user_id = df_users["user_id"].iloc[0] # on prend le 1 user
    user_data = df_users[df_users["user_id"] == user_id].iloc[0] # on extrait ses données
    possible_activities = df_activities.copy()
    possible_activities["user_id"] = user_id
    possible_activities["persona_category"] = le["persona_category"].transform([user_data["persona_category"]])[0]
    possible_activities["category"] = le["category"].transform(possible_activities["category"])
    possible_activities["location"] = le["location"].transform(possible_activities["location"])

    features = ["user_id", "activity_id", "category", "vendor_id", "likes", "rating"]
    X_new = possible_activities[features] 
    possible_activities["predicted_rating"] = best_model.predict(X_new) # Prédiction des notes que l'utilisateur donnerait à chaque activité
    recommendations = possible_activities.sort_values(by="predicted_rating", ascending=False).head(5)

    assert len(recommendations) == 5 # verifie 5 reco
    assert "predicted_rating" in recommendations.columns #colonne presente

def test_label_encoding():
    df, _, _, le = load_and_prepare_data(".")
    for col in ["persona_category", "category", "location"]:
        assert df[col].isnull().sum() == 0  # Pas de valeurs manquantes
        assert (df[col] >= 0).all()  # Toutes les valeurs sont positives (encodage correct)

def test_train_test_split_size():
    df, *_ = load_and_prepare_data(".")
    features = ["user_id", "activity_id", "category", "vendor_id", "likes", "rating"]
    target = "user_rating"
    X = df[features] #colonne
    y = df[target] #variable que l'on veut predire 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # test 20%, 42 aléatoire
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(df) # on n’a rien perdu ni dupliqué
    assert abs(len(X_test) - 0.2 * len(df)) < 2  

def test_data_merge_integrity():
    df_users = pd.read_csv("utilisateurs.csv")
    df_activities = pd.read_csv("activites.csv")
    df_interactions = pd.read_csv("interactions_utilisateur.csv")
    df = df_interactions.merge(df_users, on="user_id", how="left")
    df = df.merge(df_activities, on="activity_id", how="left")
    # Le nombre de lignes du df final doit être = interactions 
    assert len(df) == len(df_interactions)
    # Aucun user_id ou activity_id ne doit être perdu
    assert df["user_id"].isnull().sum() == 0 #case null = 0
    assert df["activity_id"].isnull().sum() == 0

