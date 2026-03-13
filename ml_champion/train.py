# %%

import mlflow

import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
from sklearn import pipeline

from feature_engine import imputation

import matplotlib.pyplot as plt

import dotenv
dotenv.load_dotenv

import os


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# %%

mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))
mlflow.set_experiment(experiment_id=677237241718907824)

# %%
df = pd.read_csv("../data/abt_f1_drivers_champion.csv", sep=";")
df.head()

# %%

### SEMMA
#### SAMPLING

df['year'] = df['dt_ref'].apply(lambda x: x.split("-")[0]).astype(int)
df_oot = df[df['year']==2025].copy()
df_oot

df_analytics = df[df['year']<2025].copy()

# %%

df_year_round = df_analytics[['year', 'dt_ref']].drop_duplicates()
df_year_round
df_year_round['row_number'] = (df_year_round.sort_values('dt_ref', ascending=False)
                                            .groupby('year')
                                            .cumcount())

df_year_round[['year','dt_ref','row_number']]

df_year_round = df_year_round[df_year_round['row_number'] > 4]
df_year_round = df_year_round.drop('row_number', axis=1)
df_year_round

# %%

df_driver_year = df_analytics[['driverid', 'year', 'flChampion']].drop_duplicates()
df_driver_year.sort_values(["driverid", "year"], ascending=[True, False])

train, test = model_selection.train_test_split(
    df_driver_year,
    random_state=42,
    train_size=0.8,
    stratify=df_driver_year['flChampion'],
)

print("Taxa de Campeões Treino:", train['flChampion'].mean())
print("Taxa de Campeões Teste:", test['flChampion'].mean())

df_train = train.merge(df_analytics).merge(df_year_round, how='inner')
df_test = test.merge(df_analytics).merge(df_year_round, how='inner')

print("Quantidade de linhas train:", df_train.shape)
print("Quantidade de linhas test:", df_test.shape)

features = df_train.columns[4:]
features

X_train, y_train =  df_train[features], df_train['flChampion']
X_test, y_test =  df_test[features], df_test['flChampion']
X_oot, y_oot =  df_oot[features], df_oot['flChampion']

# %%

#### EXPLORE

isna = X_train.isna().sum()
isna[isna > 0 ]

# %%

missing = imputation.ArbitraryNumberImputer(
    arbitrary_number=-10000,
    variables=X_train.columns.tolist())


clf = ensemble.RandomForestClassifier(
    min_samples_leaf=50,
    n_estimators=500,
    random_state=42,
    n_jobs=4,
)

model = pipeline.Pipeline(steps=[
    ('Imputation', missing),
    ("RandomForest", clf)
])

# %%

with mlflow.start_run():
    model.fit(X_train, y_train)
    
    y_train_prob = model.predict_proba(X_train)[:,1]
    roc_train = metrics.roc_curve(y_train, y_train_prob)
    auc_train = metrics.roc_auc_score(y_train, y_train_prob)
    mlflow.log_metric("ROC Train", auc_train)

    y_test_prob = model.predict_proba(X_test)[:,1]
    roc_test = metrics.roc_curve(y_test, y_test_prob)
    auc_test = metrics.roc_auc_score(y_test, y_test_prob)
    mlflow.log_metric("ROC Test", auc_test)

    y_oot_pred = model.predict(X_oot)
    y_oot_prob = model.predict_proba(X_oot)[:,1]
    auc_oot = metrics.roc_auc_score(y_oot, y_oot_prob)
    roc_oot = metrics.roc_curve(y_oot, y_oot_prob)
    mlflow.log_metric("ROC OOT", auc_oot)

    plt.figure(dpi=100)
    plt.plot(roc_train[0], roc_train[1])
    plt.plot(roc_test[0], roc_test[1])
    plt.plot(roc_oot[0], roc_oot[1])
    plt.legend([f"Treino: {auc_train:.4f}", f"Teste: {auc_test:.4f}", f"OOT: {auc_oot:.4f}"])
    plt.grid(True)
    plt.title("Curva ROC")
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")

    feature_importance = pd.Series(clf.feature_importances_, index=X_train.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    feature_importance.to_markdown("feature_importances.md")
    mlflow.log_artifact("feature_importances.md")

    model.fit(df[features], df['flChampion'])

    mlflow.sklearn.log_model(model, name="model")

