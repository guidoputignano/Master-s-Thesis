# Robustness to parametric plant-model mismatch: summary

Initial-condition seeds per plant: 10. Joint Latin hypercube samples: 64. Run length: 6 one-hour intervals. Master (aleatory) seed: 42. Perturbation (epistemic) seed: 10042. Constraint: phi_sen <= 0.3. Values are mean with a 95 percent confidence interval over the initial-condition seeds.

## Feedback efficacy over the joint Latin hypercube

| configuration | mean terminal phi_sen | max terminal phi_sen | p95 terminal phi_sen | mean J | fraction of samples with any violation |
|---|---|---|---|---|---|
| closed-loop (nominal internal model) | 0.2094 | 0.2210 | 0.2189 | 14.347 | 0.000 |
| open-loop feedforward | 0.2092 | 0.2210 | 0.2189 | 14.347 | 0.000 |

No-mismatch reference (nominal plant, closed loop): terminal phi_sen = 0.2085 +/- 0.0001, J = 13.416 +/- 0.003, violation frequency 0.000.

## One-at-a-time perturbations (tornado inputs)

| parameter | direction | value | closed terminal phi_sen | open terminal phi_sen | closed J | closed violation freq. | open violation freq. |
|---|---|---|---|---|---|---|---|
| morph | low | 5.92 | 0.2082 +/- 0.0001 | 0.2085 +/- 0.0001 | 12.191 | 0.000 | 0.000 |
| morph | high | 8.88 | 0.2087 +/- 0.0001 | 0.2085 +/- 0.0001 | 14.386 | 0.000 | 0.000 |
| gamma_min | low | 0.002224 | 0.2057 +/- 0.0001 | 0.2056 +/- 0.0001 | 13.358 | 0.000 | 0.000 |
| gamma_min | high | 0.003336 | 0.2113 +/- 0.0001 | 0.2113 +/- 0.0001 | 13.474 | 0.000 | 0.000 |
| alpha_gamma | low | 0.003976 | 0.2070 +/- 0.0001 | 0.2070 +/- 0.0001 | 13.388 | 0.000 | 0.000 |
| alpha_gamma | high | 0.005964 | 0.2099 +/- 0.0001 | 0.2099 +/- 0.0001 | 13.443 | 0.000 | 0.000 |
| tau_opt | low | 1.12 | 0.2177 +/- 0.0001 | 0.2178 +/- 0.0001 | 13.604 | 0.000 | 0.000 |
| tau_opt | high | 1.68 | 0.2030 +/- 0.0001 | 0.2030 +/- 0.0001 | 13.311 | 0.000 | 0.000 |

## Adaptation-constant sweep, 6 to 12 h (closed loop)

| adaptation constant (h) | terminal phi_sen | J | violation freq. |
|---|---|---|---|
| 6.0 | 0.2082 +/- 0.0001 | 12.265 | 0.000 |
| 7.0 | 0.2084 +/- 0.0001 | 13.114 | 0.000 |
| 8.0 | 0.2086 +/- 0.0001 | 13.835 | 0.000 |
| 9.0 | 0.2087 +/- 0.0001 | 14.455 | 0.000 |
| 10.0 | 0.2088 +/- 0.0001 | 14.993 | 0.000 |
| 11.0 | 0.2089 +/- 0.0001 | 15.462 | 0.000 |
| 12.0 | 0.2090 +/- 0.0001 | 15.876 | 0.000 |
