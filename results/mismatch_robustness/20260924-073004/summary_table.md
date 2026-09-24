# Robustness to parametric plant-model mismatch: summary

Initial-condition seeds per plant: 5. Joint Latin hypercube samples: 24. Run length: 6 one-hour intervals. Master (aleatory) seed: 42. Perturbation (epistemic) seed: 10042. Constraint: phi_sen <= 0.3. Values are mean with a 95 percent confidence interval over the initial-condition seeds.

## Feedback efficacy over the joint Latin hypercube

| configuration | mean terminal phi_sen | max terminal phi_sen | p95 terminal phi_sen | mean J | fraction of samples with any violation |
|---|---|---|---|---|---|
| closed-loop (nominal internal model) | 0.2695 | 0.2729 | 0.2719 | 13.223 | 0.000 |
| open-loop feedforward | 0.2695 | 0.2749 | 0.2731 | 13.219 | 0.000 |

No-mismatch reference (nominal plant, closed loop): terminal phi_sen = 0.2695 +/- 0.0000, J = 13.255 +/- 0.000, violation frequency 0.000.

## One-at-a-time perturbations (tornado inputs)

| parameter | direction | value | closed terminal phi_sen | open terminal phi_sen | closed J | closed violation freq. | open violation freq. |
|---|---|---|---|---|---|---|---|
| morph | low | 2.4 | 0.2693 +/- 0.0000 | 0.2695 +/- 0.0000 | 12.623 | 0.000 | 0.000 |
| morph | high | 3.6 | 0.2696 +/- 0.0000 | 0.2695 +/- 0.0000 | 13.793 | 0.000 | 0.000 |
| gamma_min | low | 0.002224 | 0.2683 +/- 0.0000 | 0.2677 +/- 0.0000 | 13.216 | 0.000 | 0.000 |
| gamma_min | high | 0.003336 | 0.2707 +/- 0.0000 | 0.2713 +/- 0.0000 | 13.295 | 0.000 | 0.000 |
| gamma_max | low | 0.01 | 0.2681 +/- 0.0000 | 0.2672 +/- 0.0000 | 13.201 | 0.000 | 0.000 |
| gamma_max | high | 0.015 | 0.2710 +/- 0.0000 | 0.2718 +/- 0.0000 | 13.312 | 0.000 | 0.000 |
| tau_h | low | 0.4 | 0.2679 +/- 0.0000 | 0.2670 +/- 0.0000 | 13.199 | 0.000 | 0.000 |
| tau_h | high | 0.6 | 0.2712 +/- 0.0000 | 0.2720 +/- 0.0000 | 13.313 | 0.000 | 0.000 |

## Adaptation-constant sweep, 2 to 8 h (closed loop)

| adaptation constant (h) | terminal phi_sen | J | violation freq. |
|---|---|---|---|
| 2.0 | 0.2692 +/- 0.0000 | 12.146 | 0.000 |
| 2.5 | 0.2694 +/- 0.0000 | 12.736 | 0.000 |
| 3.0 | 0.2695 +/- 0.0000 | 13.255 | 0.000 |
| 3.5 | 0.2696 +/- 0.0000 | 13.710 | 0.000 |
| 4.0 | 0.2697 +/- 0.0000 | 14.108 | 0.000 |
| 6.0 | 0.2699 +/- 0.0000 | 15.287 | 0.000 |
| 8.0 | 0.2701 +/- 0.0000 | 16.049 | 0.000 |
