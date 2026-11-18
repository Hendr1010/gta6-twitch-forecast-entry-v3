# GTA VI Twitch CCU Forecaster v2

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Hendr1010/gta6-twitch-forecast-entry-v3/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Hendr1010/gta6-twitch-forecast-entry-v3/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts hourly Twitch concurrent viewers (hours 2–24, UTC) for the Grand Theft Auto VI release day, as 5 quantiles (5%, 25%, 50%, 75%, 95%), given only the hour-1 print.

## Model

Local-linear log-ratio regression. For each hour `h`, the log-ratio `r_h = ln(viewers_h / viewers_1)` is regressed on `log(hour-1 viewers)` across 37 comparable console/PC launch curves, using a Gaussian-kernel-weighted local-linear fit (bandwidth 1.5 on the log scale). The size-dependence is strong: at peak hours the correlation between `log(p1)` and `r_h` is −0.94 (slope ≈ −0.9) — small hour-1 prints grow explosively, huge hour-1 prints stay flat. The v1 similarity-weighted quantile model missed this and failed badly on small launches.

The prediction band comes from leave-one-out local-fit residuals, weighted by proximity to the target's `log(p1)` (small launches are more dispersed than large ones), rescaled to a floor/cap, narrowed by 0.85, and widened when the target's hour-1 print lies outside the observed comparable range.

## Results (practice set, local pinball loss)

| Case | v1 | v2 |
| --- | --- | --- |
| p00 Wolverine | 0.3782 | 0.1588 |
| p01 Red Dead Redemption 2 | 0.1780 | 0.1731 |
| p02 God of War Ragnarök | 0.1572 | 0.1237 |
| p03 Spider-Man 2 | 0.2038 | 0.1926 |
| **Overall** | **0.2293** | **0.1621** |

Leave-one-comparable-out (37 launches): 0.410 vs v1's 0.446. House baseline: 0.387.

## Run

```sh
MAKEFUN_INPUT_DIR=<kit input dir> MAKEFUN_OUTPUT_DIR=<output dir> node main.ts
```

Node 22, standard library only, no network, deterministic, ~0.15s per case.

## License

MIT — see [LICENSE](LICENSE).
