# Cycle-loss arms vs the no-cycle baseline — Stable Audio

Validation inversion loss (L_inv only, so every arm is scored on the same quantity). Commit `7407b720`.

Figure: `cycle_arms.png`.

| step | no cycle (baseline) | k=1, ratio 0.1 | k=1, ratio 0.5 | k=1, ratio 1.0 | k=1, ratio 2.0 | k=2, ratio 1.0 | k=3, ratio 1.0 |
|---|---|---|---|---|---|---|---|
| 1000 | 2.892141e-05 | 2.892147e-05 | 2.892133e-05 | 2.892159e-05 | 2.892152e-05 | 3.017465e-05 | 3.002068e-05 |
| 2000 | 2.603649e-05 | 2.603641e-05 | 2.603650e-05 | 2.603652e-05 | 2.603647e-05 | 2.695585e-05 | — |
| 3000 | 2.521240e-05 | 2.521228e-05 | 2.521235e-05 | 2.521239e-05 | 2.521239e-05 | — | — |
| 4000 | 2.439417e-05 | 2.439419e-05 | 2.439427e-05 | 2.439425e-05 | 2.439423e-05 | — | — |
| 5000 | 2.463338e-05 | — | — | — | — | — | — |

## Spread across the four k=1 arms (20x span in target_ratio)

| step | min | max | relative spread |
|---|---|---|---|
| 1000 | 2.892133116e-05 | 2.892159192e-05 | 9.02e-06 |
| 2000 | 2.603641040e-05 | 2.603651954e-05 | 4.19e-06 |
| 3000 | 2.521227978e-05 | 2.521239423e-05 | 4.54e-06 |
| 4000 | 2.439418500e-05 | 2.439426817e-05 | 3.41e-06 |
