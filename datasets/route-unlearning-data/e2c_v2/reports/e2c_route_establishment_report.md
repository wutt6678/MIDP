# E2C Route Establishment Report

## Summary

**ROUTE_ESTABLISHED**: False
**CONTROLLED_UNLEARNING_ALLOWED**: False

## Probe Accuracy Table

| Condition | I2N | NAME | DV-syn | IPN-syn | WN | VTC | NameEffect | WrongNameEffect | ConflictEffect |
|-----------|-----|------|--------|---------|----|-----|------------|-----------------|----------------|
| M | 0.033 | 0.400 | 0.367 | 0.467 | 0.600 | 0.367 | 0.0083 | 0.0708 | -1.2375 |
| D | - | 0.500 | 0.600 | 0.600 | 0.633 | 0.367 | - | 0.1417 | -2.2750 |
| M-shuffled | - | - | - | - | - | - | - | - | - |

## Between-Condition Contrasts

- **NAME_M_minus_D**: -0.1000
- **abs_WrongNameEffect_M_minus_D**: -0.0708 (CI: [-0.1429, 0.0139])
- **abs_ConflictEffect_M_minus_D**: -1.0375 (CI: [-1.9643, -0.1726])

## M-shuffled Analysis

- NAME: true_agreement=0.5000, shuffled_agreement=0.5000
- DV_syn: true_agreement=0.4333, shuffled_agreement=0.5667
- IPN_syn: true_agreement=0.5000, shuffled_agreement=0.5000

## Gate Results (R1–R7)

- **R1**: FAIL
- **R2**: FAIL
- **R3**: FAIL
- **R4**: FAIL
- **R5**: FAIL
- **R6**: FAIL
- **R7**: FAIL

## Decision

**ROUTE_ESTABLISHED = False**

