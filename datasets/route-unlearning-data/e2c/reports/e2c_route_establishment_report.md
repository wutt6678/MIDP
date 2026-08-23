# E2C Route Establishment Report

## Summary

**ROUTE_ESTABLISHED**: False
**CONTROLLED_UNLEARNING_ALLOWED**: False

## Probe Accuracy Table

| Condition | I2N | NAME | DV-syn | IPN-syn | WN | VTC | NameEffect | WrongNameEffect | ConflictEffect |
|-----------|-----|------|--------|---------|----|-----|------------|-----------------|----------------|
| M | 0.000 | 0.600 | 0.233 | 0.500 | 0.533 | 0.500 | 0.0417 | 0.0375 | 0.0250 |
| D | - | 0.500 | 0.867 | 0.500 | 0.500 | 0.500 | - | 0.0792 | -0.7708 |
| M-shuffled | - | - | - | - | - | - | - | - | - |

## Between-Condition Contrasts

- **NAME_M_minus_D**: 0.1000
- **abs_WrongNameEffect_M_minus_D**: -0.0417 (CI: [-0.1458, 0.0119])
- **abs_ConflictEffect_M_minus_D**: -0.7458 (CI: [-1.0833, 0.8611])

## M-shuffled Analysis

- NAME: true_agreement=0.5000, shuffled_agreement=0.5000
- DV_syn: true_agreement=0.6333, shuffled_agreement=0.3667
- IPN_syn: true_agreement=0.4667, shuffled_agreement=0.5333

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

