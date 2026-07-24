# Robustness to parametric plant-model mismatch: summary

Initial-condition seeds per plant: 5. Joint Latin hypercube samples: 24. Run length: 6 one-hour intervals. Master (aleatory) seed: 42. Perturbation (epistemic) seed: 10042. Constraint: phi_sen <= 0.3. Values are mean with a 95 percent confidence interval over the initial-condition seeds.

## Feedback efficacy over the joint Latin hypercube

| configuration | mean terminal phi_sen | max terminal phi_sen | p95 terminal phi_sen | mean J | fraction of samples with any violation |
|---|---|---|---|---|---|
| closed-loop (nominal internal model) | 0.2721 | 0.2753 | 0.2743 | 14.927 | 0.000 |
| open-loop feedforward | 0.2720 | 0.2773 | 0.2755 | 14.927 | 0.000 |

No-mismatch reference (nominal plant, closed loop): terminal phi_sen = 0.2720 +/- 0.0000, J = 14.337 +/- 0.000, violation frequency 0.000.

## One-at-a-time perturbations (tornado inputs)

| parameter | direction | value | closed terminal phi_sen | open terminal phi_sen | closed J | closed violation freq. | open violation freq. |
|---|---|---|---|---|---|---|---|
| morph | low | 5.92 | 0.2719 +/- 0.0000 | 0.2720 +/- 0.0000 | 13.519 | 0.000 | 0.000 |
| morph | high | 8.88 | 0.2721 +/- 0.0000 | 0.2720 +/- 0.0000 | 14.961 | 0.000 | 0.000 |
| gamma_min | low | 0.002224 | 0.2709 +/- 0.0000 | 0.2702 +/- 0.0000 | 14.307 | 0.000 | 0.000 |
| gamma_min | high | 0.003336 | 0.2731 +/- 0.0000 | 0.2738 +/- 0.0000 | 14.368 | 0.000 | 0.000 |
| gamma_max | low | 0.01 | 0.2707 +/- 0.0000 | 0.2698 +/- 0.0000 | 14.299 | 0.000 | 0.000 |
| gamma_max | high | 0.015 | 0.2734 +/- 0.0000 | 0.2743 +/- 0.0000 | 14.379 | 0.000 | 0.000 |
| tau_h | low | 0.4 | 0.2705 +/- 0.0000 | 0.2696 +/- 0.0000 | 14.297 | 0.000 | 0.000 |
| tau_h | high | 0.6 | 0.2735 +/- 0.0000 | 0.2745 +/- 0.0000 | 14.380 | 0.000 | 0.000 |

## Adaptation-constant sweep, 6 to 12 h (closed loop)

| adaptation constant (h) | terminal phi_sen | J | violation freq. |
|---|---|---|---|
| 6.0 | 0.2719 +/- 0.0000 | 13.569 | 0.000 |
| 7.0 | 0.2720 +/- 0.0000 | 14.139 | 0.000 |
| 8.0 | 0.2721 +/- 0.0000 | 14.610 | 0.000 |
| 9.0 | 0.2721 +/- 0.0000 | 15.006 | 0.000 |
| 10.0 | 0.2722 +/- 0.0000 | 15.342 | 0.000 |
| 11.0 | 0.2722 +/- 0.0000 | 15.632 | 0.000 |
| 12.0 | 0.2722 +/- 0.0000 | 15.884 | 0.000 |
