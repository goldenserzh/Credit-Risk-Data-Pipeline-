from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
import os
import psycopg2
from airflow.hooks.base import BaseHook
from src.preprocessing import (numcat_columns,
                           fill_missing_categor,
                           fill_missing_numeric,
                           calculate_total_debt,
                           calculate_dti,
                           calculate_previous_credits,
                           child_classification,
                           family_feature)
def get_bd_connection_engine():
    try:
        if 'prod_bogdan' in os.getcwd() or 'app' in os.getcwd():
            HOST = os.getenv("HOST", '82.202.137.136')
            PG_LOGIN = os.getenv("PG_LOGIN", 'sgoldaev')
            PG_PASS = os.getenv("PG_PASS", 'y4Q1dBE0Y53c0dwe')
            PG_DB = os.getenv("PG_DB", 'dbeaver')
        else:
            pg_con = BaseHook.get_connection("postgres_default")
            HOST = pg_con.host
            PG_LOGIN = pg_con.login
            PG_PASS = pg_con.password
            PG_DB = pg_con.schema

        conn_string = f"postgresql+psycopg2://{PG_LOGIN}:{PG_PASS}@{HOST}:5432/{PG_DB}"
        engine = create_engine(conn_string)
        return engine
    except Exception as e:
        print(f'Ошибка при подключении {e}')

    

def load_data_sets(**context):
    try:
        engine = get_bd_connection_engine()
        df_train = pd.read_sql("SELECT * FROM dwh.application_train", engine, chunksize=10000)
        df_test = pd.read_sql("SELECT * FROM dwh.application_test", engine, chunksize=10000)
        bureau = pd.read_sql("SELECT * FROM FROM dwh.bureau", engine, chunksize=10000)
        
        df_train = pd.concat(df_train)
        df_test = pd.concat(df_test)
        bureau = pd.concat(bureau)
        print("Данные загрузились!")

        context['ti'].xcom_push(key='df_train', value=df_train.to_json()) 
        context['ti'].xcom_push(key='df_test', value=df_test.to_json())
        context['ti'].xcom_push(key='bureau', value=bureau.to_json())
    except Exception as e:
        raise Exception(f"Ошибка загрузки данных: {str(e)}")
        

def preprocessing_df_and_load(**context):
    try:
        df_train = pd.read_json(context['ti'].xcom_pull(key='df_train'))
        df_test = pd.read_json(context['ti'].xcom_pull(key='df_test'))
        bureau = pd.read_json(context['ti'].xcom_pull(key='bureau'))

        train_categor_feature, train_num_feature = numcat_columns(df_train) 
        test_categor_feature, test_num_feature = numcat_columns(df_test)
        
        df_train = fill_missing_categor(df_train, train_categor_feature)
        df_test = fill_missing_categor(df_test, test_categor_feature)

        fill_missing_numeric(df_train, train_num_feature)
        fill_missing_numeric(df_test, test_num_feature)

        df_train, df_test = calculate_total_debt(df_train, df_test, bureau)
        df_train, df_test = calculate_dti(df_train, df_test)
        df_train, df_test = calculate_previous_credits(df_train, df_test, bureau)
        df_train, df_test = family_feature(df_train, df_test)

        engine = get_bd_connection_engine()
        df_train.to_sql('application_train_processed', engine, schema='dwh', if_exists='replace', index=False)
        df_test.to_sql('application_test_processed', engine, schema='dwh', if_exists='replace', index=False)

        print("Предобработка завершена, данные сохранены!")
    except Exception as e:
        raise Exception (f"Не удалось преобразовать данные: {str(e)}")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
}

with DAG(
    dag_id='task_1',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    default_args=default_args
) as dag:
    load_dfs_task = PythonOperator(
        task_id = 'load_dfs',
        python_callable=load_data_sets,
        provide_context = True
    )

    preprocessing_df = PythonOperator(
        task_id = 'preprocessing',
        python_callable=preprocessing_df_and_load, 
        procide_context = True
    )



load_dfs_task >> preprocessing_df
