"""Statistics engine. Pure functions: no database, no network."""

from .agreement import AgreementReport, binary_agreement, cohen_kappa, weighted_kappa
from .core import bootstrap_mean_ci, case_means, holm_adjust, mcnemar_exact, percentile, sign_flip_pvalue
from .flaky import FlakyCase, flakiness_score, flaky_cases, is_flaky
from .power import data_multiplier, mde, null_difference_sd
from .verdict import MIN_CASES, RunView, VerdictReport, evaluate, metric_value

__all__ = [
    "AgreementReport",
    "FlakyCase",
    "MIN_CASES",
    "RunView",
    "VerdictReport",
    "binary_agreement",
    "bootstrap_mean_ci",
    "case_means",
    "cohen_kappa",
    "data_multiplier",
    "evaluate",
    "flakiness_score",
    "flaky_cases",
    "holm_adjust",
    "is_flaky",
    "mcnemar_exact",
    "mde",
    "metric_value",
    "null_difference_sd",
    "percentile",
    "sign_flip_pvalue",
    "weighted_kappa",
]
