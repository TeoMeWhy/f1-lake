# %%
import flask
import mlflow
import pandas as pd

import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file

import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))

models = mlflow.search_registered_models(filter_string="name='f1_driver_champion'")[-1]
last_version = max([int(i.version) for i in models.latest_versions])
MODEL = mlflow.sklearn.load_model(f"models:/f1_driver_champion/{last_version}")


# %%

app = flask.Flask(__name__)

@app.route('/health_check')
def health_check():
    return "OK", 200


@app.route('/predict', methods=['POST'])
def predict():
    payload = flask.request.get_json()
    data = payload.get('values', [])
    if len(data) == 0:
        return {"error": "No features provided"}, 400

    df = pd.DataFrame(data)
    X = df[MODEL.feature_names_in_]
    
    
    df_proba = pd.DataFrame(MODEL.predict_proba(X), columns=MODEL.classes_)
    df_proba['id'] = df['id'].copy()
    df_proba.set_index('id', inplace=True)

    payload = df_proba.to_dict(orient='index')

    return {"predictions": payload}, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)