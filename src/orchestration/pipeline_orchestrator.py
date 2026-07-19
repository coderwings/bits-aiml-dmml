"""
End-to-End Modular Pipeline Orchestrator for RecoMart Recommendation System.
Orchestrates: Ingestion -> Validation -> Preparation -> Feature Engineering -> Feature Store -> Model Training -> Data Versioning.
"""

import os
import time
import logging
import traceback
from datetime import datetime

from src.ingestion.data_ingestion import run_ingestion
from src.validation.data_validation import run_validation
from src.preparation.data_preparation import run_preparation
from src.features.feature_engineering import run_feature_engineering
from src.features.feature_store import RecoMartFeatureStore
from src.versioning.data_versioning import run_versioning
from src.model.train_evaluation import run_model_training
from src.model.inference import RecommendationInferenceService
from src.reports.pdf_generator import generate_problem_formulation_pdf

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

logger = logging.getLogger("RecoMart_Orchestration")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('logs/pipeline_orchestration.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


class PipelineOrchestrator:
    def __init__(self):
        self.stage_results = {}
        self.pipeline_start_time = None

    def execute_stage(self, stage_name, func, *args, **kwargs):
        logger.info(f"--------------------------------------------------")
        logger.info(f"PIPELINE STAGE START: {stage_name}")
        logger.info(f"--------------------------------------------------")
        t0 = time.time()
        try:
            result = func(*args, **kwargs)
            duration = round(time.time() - t0, 2)
            logger.info(f"STAGE SUCCESS: '{stage_name}' completed in {duration} seconds.")
            self.stage_results[stage_name] = {"status": "SUCCESS", "duration_sec": duration, "output": result}
            return result
        except Exception as e:
            duration = round(time.time() - t0, 2)
            logger.error(f"STAGE FAILED: '{stage_name}' failed after {duration} seconds: {str(e)}")
            logger.error(traceback.format_exc())
            self.stage_results[stage_name] = {"status": "FAILED", "duration_sec": duration, "error": str(e)}
            raise e

    def run_full_pipeline(self):
        self.pipeline_start_time = datetime.now()
        logger.info(f"Starting RecoMart Data Management & ML Pipeline Execution at {self.pipeline_start_time}")

        # 1. Problem Formulation Deliverable Generation
        self.execute_stage("01_Problem_Formulation", generate_problem_formulation_pdf)

        # 2. Data Ingestion Stage
        self.execute_stage("02_Data_Ingestion", run_ingestion)

        # 3. Data Validation & Quality Profiling Stage
        self.execute_stage("03_Data_Validation", run_validation)

        # 4. Data Preparation & Cleaning Stage
        self.execute_stage("04_Data_Preparation", run_preparation)

        # 5. Feature Engineering Stage
        self.execute_stage("05_Feature_Engineering", run_feature_engineering)

        # 6. Feature Store Sync Stage
        self.execute_stage("06_Feature_Store_Sync", lambda: RecoMartFeatureStore())

        # 7. Data Versioning and Lineage Stage
        self.execute_stage("07_Data_Versioning", run_versioning)

        # 8. Model Training & Evaluation Stage
        self.execute_stage("08_Model_Training_Evaluation", run_model_training)

        # 9. Real-Time Inference Warm-up & Verification
        def test_inference():
            service = RecommendationInferenceService()
            return service.get_recommendations("1001", top_k=5)
        self.execute_stage("09_Inference_Verification", test_inference)

        total_duration = round((datetime.now() - self.pipeline_start_time).total_seconds(), 2)
        logger.info(f"RECOMART PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {total_duration} SECONDS")
        return self.stage_results


def main():
    orchestrator = PipelineOrchestrator()
    orchestrator.run_full_pipeline()

if __name__ == "__main__":
    main()
