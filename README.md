# Adaptive Edge–Cloud Scheduling for Machine Learning Inference

## Research problem

ML inference can execute on resource-constrained edge nodes or on a higher-capacity cloud node. 
Edge execution can reduce network latency and modeled cost, but limited compute capacity can cause queueing and missed deadlines 
as demand increases. Cloud execution provides more compute capacity but incurs higher modeled latency and per-request cost. 
In this project, we investigates how scheduling policy, workload intensity, and network variability affect latency, resource utilization, cost, and deadline violations.

The study compares three deterministic baselines with an XGBoost-based scheduler:

- **Always Edge** — dispatches requests to an edge node.
- **Always Cloud** — dispatches requests to the cloud.
- **Greedy** — selects the lowest-estimated-latency node that can handle the request.
- **XGBoost** — predicts the execution node from synthetic training data.

The central comparison is whether XGBoost improves on a capacity-aware heuristic, ad not merely whether adaptive placement improves on a fixed placement policy.

## Architecture

The project has two complementary components:

1. **Controlled simulator:** evaluates the 4 scheduling policies across workload and network scenarios using synthetic requests and repeatable random seeds.
2. **Real-inference extension:** uses MobileNetV3 Small for image classification, a FastAPI `/predict` endpoint, Docker, and an Amazon EC2 deployment running Amazon Linux 2023. 
Simulator metrics and real HTTP/inference timings are different measurements and should not be conflated.

### Simulated nodes

| Node | Compute capacity | Base network latency | Modeled cost per request |
|---|---:|---:|---:|
| `edge_1` | 10 | 5 ms | 0.010 |
| `edge_2` | 20 | 10 ms | 0.015 |
| `cloud` | 100 | 50 ms | 0.050 |

The costs are simulator-defined units, not measured AWS charges. Latencies are simulator parameters, not measurements from the EC2 deployment.

### Experimental design

| Setting | Configuration |
|---|---|
| Training data | 10,000 synthetic samples |
| Evaluation | 1,000 requests per run |
| Seeds | 1–5 |
| Network conditions | `stable`, `moderate`, `high` |
| Workload levels | `low_load`, `medium_load`, `high_load`, `very_high_load` |
| Arrival intervals | 50, 20, 10, 5, respectively |
| Network variation | Stable: fixed base latency; moderate: factor 0.8–1.2; high: factor 0.5–2.0 |
| Network updates | Every 10 requests |
| Metrics | Mean ± standard deviation of latency, cost, utilization, and deadline-violation rate |

The experiment script reports results aggregated over five seeds. The values below reproduce the supplied output from `experiments/core_experiments/run_comparison.py`.

## Repository structure

```text
edge-cloud-inference-scheduler/
├── nodes/
│   └── node.py                         # Node model and centralized node creation
├── simulation/
│   ├── workload.py                     # Synthetic request generation
│   └── runner.py                       # Policy execution and simulation
├── scheduler/
│   └── baselines.py                    # Always Edge, Always Cloud, Greedy
├── ml/
│   ├── dataset.py                      # Synthetic training-data generation
│   └── xgboost_scheduler.py             # XGBoost scheduler
├── evaluation/
│   └── results.py                      # Evaluation metrics/result handling
├── experiments/
│   ├── run_baseline.py                 # Baseline experiment entry point
│   ├── core_experiments/
│   │   └── run_comparison.py           # Main policy comparison
│   └── plots/
│       ├── plot_results.py             # Research figures
│       └── figures/                    # Generated figures
└── README.md
```

The tree lists the principal modules used in the implementation; minor files may exist beyond those shown.

## XGBoost configuration

The current recorded configuration is:

| Parameter | Value |
|---|---:|
| `n_estimators` | 300 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `min_child_weight` | 3 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.8 |
| `gamma` | 0.3 |
| `reg_lambda` | 1 |

The scheduler is trained to predict a target execution node from synthetic examples. It should be interpreted as a supervised node-selection model, not as a latency-regression model.

## Results

All values in this section are from the supplied run. Values are **mean ± standard deviation across five seeds**. Latency is in milliseconds; cost is in simulator-defined units; utilization and deadline violations are percentages. Values are rounded to two decimal places.

### Very-high workload: complete policy comparison

| Network | Policy | Latency (ms) | Cost | Utilization | Deadline violations |
|---|---|---:|---:|---:|---:|
| Stable | Always Edge | 27,101.39 ± 263.03 | 10.00 ± 0.00 | 99.94 ± 0.04% | 99.50 ± 0.14% |
| Stable | Always Cloud | 50.06 ± 0.00 | 50.00 ± 0.00 | 15.19 ± 0.54% | 0.06 ± 0.05% |
| Stable | Greedy | 7.27 ± 0.06 | 11.73 ± 0.06 | 64.47 ± 0.95% | 0.00 ± 0.00% |
| Stable | XGBoost | 7.32 ± 0.08 | 11.69 ± 0.03 | 65.32 ± 0.62% | 0.00 ± 0.00% |
| Moderate | Always Edge | 27,077.57 ± 269.32 | 10.00 ± 0.00 | 99.90 ± 0.09% | 99.42 ± 0.22% |
| Moderate | Always Cloud | 49.73 ± 0.65 | 50.00 ± 0.00 | 17.31 ± 0.97% | 1.54 ± 0.15% |
| Moderate | Greedy | 7.11 ± 0.04 | 11.57 ± 0.05 | 59.76 ± 0.77% | 0.00 ± 0.00% |
| Moderate | XGBoost | 7.15 ± 0.05 | 11.54 ± 0.05 | 60.54 ± 0.74% | 0.00 ± 0.00% |
| High | Always Edge | 26,963.91 ± 429.84 | 10.00 ± 0.00 | 99.78 ± 0.25% | 99.12 ± 0.66% |
| High | Always Cloud | 61.35 ± 2.44 | 50.00 ± 0.00 | 35.16 ± 2.71% | 10.40 ± 1.41% |
| High | Greedy | 8.59 ± 0.08 | 11.81 ± 0.11 | 61.32 ± 1.15% | 0.00 ± 0.00% |
| High | XGBoost | 9.10 ± 0.94 | 11.94 ± 0.39 | 61.80 ± 1.52% | 0.20 ± 0.45% |

### Low-to-high workload observations

At low and medium workload, Always Edge, Greedy, and XGBoost return the same results in stable and moderate network conditions:

| Network | Policy | Latency (ms) | Cost | Utilization | Deadline violations |
|---|---|---:|---:|---:|---:|
| Stable | Edge / Greedy / XGBoost | 5.60 ± 0.00 | 10.00 ± 0.00 | 60.09 ± 0.46% | 0.00 ± 0.00% |
| Moderate | Edge / Greedy / XGBoost | 5.60 ± 0.04 | 10.00 ± 0.00 | 60.09 ± 0.46% | 0.00 ± 0.00% |

At high network variability and low/medium workload:

| Policy | Latency (ms) | Cost | Utilization | Deadline violations |
|---|---:|---:|---:|---:|
| Always Edge | 6.85 ± 0.14 | 10.00 ± 0.00 | 60.09 ± 0.46% | 0.00 ± 0.00% |
| Always Cloud | 61.32 ± 2.43 | 50.00 ± 0.00 | 7.71% at low load; 12.02% at medium load | 10.40 ± 1.41% |
| Greedy | 6.63 ± 0.10 | 10.57 ± 0.17 | 56.68 ± 0.99% | 0.00 ± 0.00% |
| XGBoost | 6.64 ± 0.10 | 10.46 ± 0.14 | 57.20 ± 0.90% | 0.00 ± 0.00% |

At high network variability and high workload:

| Policy | Latency (ms) | Cost | Utilization | Deadline violations |
|---|---:|---:|---:|---:|
| Always Edge | 19,848.44 ± 5,506.52 | 10.00 ± 0.00 | 96.18 ± 5.06% | 89.52 ± 13.14% |
| Always Cloud | 61.32 ± 2.43 | 50.00 ± 0.00 | 19.68 ± 1.68% | 10.40 ± 1.41% |
| Greedy | 6.83 ± 0.09 | 10.72 ± 0.16 | 56.40 ± 0.77% | 0.00 ± 0.00% |
| XGBoost | 6.84 ± 0.09 | 10.63 ± 0.14 | 56.85 ± 0.63% | 0.00 ± 0.00% |

### Main things that we found

1. **Fixed edge placement fails under overload.** At very-high load, Always Edge reaches 26,900–27,100 ms mean latency and 99.1–99.5% deadline violations across network conditions. Utilization is approximately 99.8–99.9%.
2. **Cloud-only placement avoids most overload-related latency but has higher modeled cost.** Its cost is 50.00 per run versus approximately 11.54–11.94 for the adaptive policies at very-high load. Under high network variability, cloud-only placement has 10.40% deadline violations.
3. **Greedy is the strongest observed policy in these experiments.** At very-high load, its mean latency is 7.11–8.59 ms, and its deadline-violation rate is zero in all three network conditions.
4. **XGBoost is competitive but does not consistently outperform Greedy.** At very-high load, XGBoost latency is 7.15–9.10 ms. It has slightly lower cost than Greedy under stable and moderate conditions, but higher cost and a 0.20% violation rate under high variability.
5. **The value of ML remains unproven relative to a strong heuristic.** In stable and moderate low-to-high workloads, XGBoost and Greedy often match exactly. Differences at very-high load are small compared with the advantage both have over fixed policies.

These are findings from the current synthetic simulator and its configured parameters. They do not establish general performance on production edge/cloud systems.

## Reproducibility

From the repository root, activate the project's virtual environment and run:

```bash
python experiments/core_experiments/run_comparison.py
```

The supplied run completed with exit code `0` and saved figures under `experiments/plots/figures/`.

The supplied output also reported two skipped plots:

- Paired latency plot: run-level results did not contain all required fields (`network_condition`, `load_level`, `policy`, `seed`, `latency`).
- Node-selection plot: node-selection counts were not included in the experiment results.

Therefore, those two plots were **not generated** by that run. To enable them, retain per-seed observations in the comparison output and include node-selection counts in the result data.

For a reproducible report, preserve the configuration, seed list, and raw per-seed results alongside generated figures. The reported standard deviations should be computed over the five seed-level outcomes, not over individual requests unless explicitly stated.

## Technologies and main libraries

- **Python 3.14** — simulator and experiment orchestration
- **XGBoost** — supervised scheduler
- **NumPy / pandas** — numerical data and result handling
- **Matplotlib**, **Seaborn** — plots
- **PyTorch / Torchvision** — MobileNetV3 Small inference
- **FastAPI** — inference HTTP endpoint
- **Docker** — containerization
- **Amazon EC2 / Amazon Linux 2023** — cloud inference deployment

## Some Limitations to be Mentioned

- The scheduler comparison uses synthetic training data and simulated node/network parameters.
- The cost values are relative simulator units, not real infrastructure billing.
- The results depend on the simulator's queueing, capacity, latency, and deadline models.
- The current evidence shows that adaptive, capacity-aware placement is substantially better than fixed placement under overload; it does not show a consistent XGBoost advantage over Greedy.
- The paired latency and node-selection plots were skipped in the supplied run because the required result fields were absent.
- Real MobileNet/EC2 measurements should be reported separately and only after their collection procedure and deployment status are verified.
