# %%

import requests
import sqlalchemy
import os
import dotenv
import pandas as pd

import streamlit as st

print(dotenv.load_dotenv())

PALANTIR_URI = os.getenv("PALANTIR_URI")

MYSQL_URI = os.getenv("MYSQL_URI")
engine = sqlalchemy.create_engine(f"mysql+pymysql://{MYSQL_URI}" )

class Driver:

    def __init__(self, driverid, driver_name):
        self.driverid = driverid
        self.driver_name = driver_name


# %%

def format_color(x:str):
    if x is None:
        return "#ffffff"

    if x.startswith("#"):
        return x.lower()
    
    return f"#{x}".lower()

def get_id_predictions(fs_ids):
    
    data = {
        "model_name": "f1_driver_champion",
        "ids": fs_ids
    }

    resp =requests.post(f"{PALANTIR_URI}/predictions", json=data)
    return resp.json().get("predictions")


@st.cache_resource(ttl="1d")
def get_predictions():
    ids = pd.read_sql("SELECT * FROM feature_store.f1_driver", engine)
    data = get_id_predictions(ids['id'].tolist())
    df = pd.DataFrame(data).T.reset_index().rename(columns={"index":"id", "1": "prob_win"})
    df = df.merge(ids, on='id')
    df['year'] = pd.to_datetime(df['dt_ref']).dt.year
    df['teamcolor'] = df['teamcolor'].apply(format_color)

    unique_drivers_name = (df[['driverid', 'dt_ref', 'fullname']].dropna()
                                                                 .sort_values(by=['driverid', 'dt_ref'])
                                                                 .drop_duplicates(subset='driverid', keep='first')
                                                                 .drop('dt_ref', axis=1)
                                                                 .rename(columns={"fullname":"fullname_correct"})
                                                              )

    df = df.merge(unique_drivers_name, on='driverid')
    return df

# %%

st.set_page_config(page_title="F1 Data App", page_icon=":racing_car:", layout="wide")
st.markdown("""
## F1 Data App :checkered_flag:

Aplicação para inferência de campeões de temporadas da Fórmula 1.

Projeto criado ao vivo no canal [Téo me Why](https://www.youtube.com/playlist?list=PLvlkVRRKOYFQBB62gy1waP3H1btp8Ll0r).

Repositório do código: [github.com/TeoMeWhy/f1-lake](https://github.com/TeoMeWhy/f1-lake)
""")

data = get_predictions()


drivers_data = (data[['driverid', 'fullname_correct']].sort_values(["driverid", "fullname_correct"])
                                                      .drop_duplicates(subset=['driverid'], keep='first')
                                                      .dropna())

drivers = [Driver(i['driverid'], i['fullname_correct']) for i in drivers_data.to_dict(orient='records')]

most_prob = (data[data['dt_ref']==data['dt_ref'].max()].sort_values(by='prob_win', ascending=False)
                                                       .head(3))

drivers_default = [i for i in drivers if i.driverid in most_prob['driverid'].tolist()]


driver_selected = st.multiselect("Pilotos",
                                 options=drivers,
                                 format_func=lambda x: x.driver_name,
                                 default=drivers_default,
                                 )


year_selected = st.multiselect("Temporada",
                               options=data['year'].unique(),
                               default=data['year'].max(),
                               )


data_filter = data[data["driverid"].isin([i.driverid for i in driver_selected])]
data_filter = data_filter[data_filter["year"].isin(year_selected)]


colors = (data_filter[['fullname_correct','dt_ref','teamcolor']].sort_values(by=['fullname_correct', 'dt_ref'], ascending=[True, False])
                                                                .drop_duplicates(subset=['fullname_correct'], keep="first")
                                                        ['teamcolor'].tolist())


data_chart = (data_filter.pivot_table(index='dt_ref', columns='fullname_correct', values='prob_win')
                         .reset_index())


column_config={i: st.column_config.NumberColumn(i, format="percent") for i in data_chart.columns[1:]}
column_config["dt_ref"] = st.column_config.DateColumn("Data Predição")


graphs, tables = st.tabs(["Gráfico", "Tabelas"])


with graphs:
    st.line_chart(data_chart,
                x='dt_ref',
                y=data_chart.columns.tolist()[1:],
                y_label="Prob. Vitória Campeonato",
                x_label='Data Pós Corrida',
                color=colors,
                )


with tables:
    st.dataframe(data_chart, column_config=column_config)

    st.markdown("Analytical Base Table")
    st.dataframe(data_filter)