import requests

url = "http://127.0.0.1:5000/recommend"
headers = {"Content-Type": "application/json"}
data = {
    "user_id": 42 
}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print("Recommandations reçues :")
    for rec in response.json():
        print(f"- {rec['name']} (Note prédite : {rec['predicted_rating']:.2f})")
else:
    print(f" Erreur {response.status_code} : {response.text}")

