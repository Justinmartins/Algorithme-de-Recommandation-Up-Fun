# Flask est un micro-framework web en Python qui permet de créer des API REST

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
df_users = pd.read_csv("utilisateurs.csv")
df_activities = pd.read_csv("activites.csv")
df_interactions = pd.read_csv("interactions_utilisateur.csv")

# Fusionner les données
df = df_interactions.merge(df_users, on="user_id", how="left")
df = df.merge(df_activities, on="activity_id", how="left")

# Encodage
label_encoders = {}
for col in ["persona_category", "category", "location"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

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

best_model_name = min(results, key=results.get)
best_model = trained_models[best_model_name]
print(f"Modèle sélectionné : {best_model_name}")

# Route API 
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400

    # Récupérer les données utilisateur
    user_rows = df_users[df_users["user_id"] == user_id]
    if user_rows.empty:
        return jsonify({"error": "user_id not found"}), 404

    user_data = user_rows.iloc[0]

    possible_activities = df_activities.copy()
    possible_activities["user_id"] = user_id

    # Transformer les colonnes catégorielles avec label_encoders
    possible_activities["persona_category"] = label_encoders["persona_category"].transform(
        [user_data["persona_category"]])[0]
    possible_activities["category"] = label_encoders["category"].transform(possible_activities["category"])
    possible_activities["location"] = label_encoders["location"].transform(possible_activities["location"])

    X_new = possible_activities[features]
    possible_activities["predicted_rating"] = best_model.predict(X_new)

    recommendations = possible_activities.sort_values(by="predicted_rating", ascending=False)[
        ["name", "predicted_rating"]].head(5)

    return jsonify(recommendations.to_dict(orient='records'))


@app.get("/")
def root():
    return {"message" : " Bienvenue sur l'API"}
    
# Pour que Flask démarre
if __name__ == "__main__":
    print("Lancement du serveur Flask...")
    app.run(debug=True)
