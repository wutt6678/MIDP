"""E2C controlled route establishment module.

Provides synthetic identity generation, condition-specific dataset building,
route probe construction, route metrics, and validation logic for the
E2C mediated/direct/shuffled route experiment.

Modules:
    synthetic_manifest — identity, alias, mapping, and split generation
    dataset_builder    — condition-specific training record generation
    training_dataset   — PyTorch Dataset for E2C training
    probe_builder      — route probe family construction
    route_metrics      — NameEffect, WrongNameEffect, ConflictEffect, bootstrap
    route_validation   — leakage checks and R1–R7 gate logic
    provenance         — full provenance capture for E2C artifacts
"""
