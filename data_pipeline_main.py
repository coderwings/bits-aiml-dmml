"""
Master Entrypoint for RecoMart Recommendation System Data Management Pipeline.
Orchestrates: Ingestion -> Validation -> Preparation -> Feature Store -> Data Versioning -> Model Training -> Inference.
"""

import sys
import os

# Auto-re-execute using ./venv/bin/python if launched with system python3
venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "venv", "bin", "python"))
if os.path.exists(venv_python) and sys.executable != venv_python:
    import subprocess
    sys.exit(subprocess.call([venv_python] + sys.argv))

import logging
from src.orchestration.pipeline_orchestrator import PipelineOrchestrator

def main():
    print("================================================================")
    print("   RecoMart Data Management & Recommendation Pipeline Execution  ")
    print("================================================================")
    
    orchestrator = PipelineOrchestrator()
    try:
        results = orchestrator.run_full_pipeline()
        print("\n================================================================")
        print("                 PIPELINE STAGE SUMMARY TABLE                  ")
        print("================================================================")
        print(f"{'Stage Name':<30} | {'Status':<10} | {'Duration (s)':<12}")
        print("-" * 60)
        for stage, meta in results.items():
            status_text = "PASS" if meta['status'] == "SUCCESS" else "FAIL"
            print(f"{stage:<30} | {status_text:<10} | {meta['duration_sec']:<12}")
        print("================================================================")
        print("\nAll deliverables generated in 'reports/', 'docs/', and 'models/' directories.")
    except Exception as e:
        print(f"\nPipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
