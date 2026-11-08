"""
pwlab.eval: Evaluation pipelines, benchmarking, and visualization.
"""

from pwlab.eval.model_eval import evaluate_test_set_coverage, run_evaluation_cli
from pwlab.eval.meter_eval import run_case_studies, run_meter_benchmark
from pwlab.eval.plots import generate_all_figures

__all__ = [
    "evaluate_test_set_coverage",
    "run_evaluation_cli",
    "run_case_studies",
    "run_meter_benchmark",
    "generate_all_figures",
]

