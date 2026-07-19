"""
Apache Airflow DAG Definition for RecoMart Data Management Pipeline.
Defines scheduled workflow tasks: Ingestion -> Validation -> Preparation -> Feature Store -> Model Training -> Evaluation.
"""

from datetime import datetime, timedelta

# Import Python callables from RecoMart pipeline modules
from src.ingestion.data_ingestion import run_ingestion
from src.validation.data_validation import run_validation
from src.preparation.data_preparation import run_preparation
from src.features.feature_engineering import run_feature_engineering
from src.versioning.data_versioning import run_versioning
from src.model.train_evaluation import run_model_training

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    default_args = {
        'owner': 'recomart_data_team',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email': ['alerts@recomart.com'],
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    }

    dag = DAG(
        'recomart_recommendation_pipeline',
        default_args=default_args,
        description='Automated RecoMart Recommendation System Data & ML Pipeline',
        schedule_interval='@daily',
        catchup=False
    )

    t1_ingestion = PythonOperator(
        task_id='ingest_raw_data',
        python_callable=run_ingestion,
        dag=dag,
    )

    t2_validation = PythonOperator(
        task_id='validate_data_quality',
        python_callable=run_validation,
        dag=dag,
    )

    t3_preparation = PythonOperator(
        task_id='clean_and_prepare_data',
        python_callable=run_preparation,
        dag=dag,
    )

    t4_feature_engineering = PythonOperator(
        task_id='feature_engineering_and_store',
        python_callable=run_feature_engineering,
        dag=dag,
    )

    t5_versioning = PythonOperator(
        task_id='data_versioning_and_lineage',
        python_callable=run_versioning,
        dag=dag,
    )

    t6_model_training = PythonOperator(
        task_id='train_and_evaluate_recommender',
        python_callable=run_model_training,
        dag=dag,
    )

    # Define DAG Dependencies Flow
    t1_ingestion >> t2_validation >> t3_preparation >> t4_feature_engineering >> t5_versioning >> t6_model_training

except ImportError:
    # Fallback if Apache Airflow is not installed in local environment
    pass
