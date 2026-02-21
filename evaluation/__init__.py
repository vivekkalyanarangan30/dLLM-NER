"""Evaluation modules for DiffusionNER-Zero.

Submodules
----------
load_benchmarks
    Download and parse CrossNER, MIT Movie, and MIT Restaurant benchmarks.
evaluate
    Entity-level micro-F1 with strict (type + text) matching.
pareto_curve
    F1 vs inference speed Pareto analysis and plotting.
self_correction
    Track entity predictions across denoising steps.
uncertainty
    Multi-run agreement and uncertainty quantification.
ablations
    Systematic ablation experiments.
speed_benchmark
    Inference throughput benchmarking.
"""
