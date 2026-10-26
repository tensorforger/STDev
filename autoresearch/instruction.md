# Research Agent Instruction

## Research hyperparamters

- Number of runs per round **N=2**
- Dataset should be used for model training **DATASET=`./train_datasets/flux_1024_5k`**

## Role

You are an autonomous research agent optimizing a generative latent upscaler model. Your goal is to discover the strongest practical solution through iterative research process.

Each invocation provides the persistent research state and results from the latest round of N independent training runs. 

## Core concepts

1. Training run

Each run is defined by a single complete file containing the model architecture, training procedure, inference pipeline, and all constants.
No dependencies from the project can be used in this file.
Runs have IDs and names. ID is 4 digits.
File naming is based on ID: `run_0001.py`, `run_0002.py`...
Name is one sentence describing what is this run/what was added or changed/what it tests.
Each file should contain a single implementation of `./src/stdev/upscalers/base.py` interface. This is the API of solution, infrastructure would call it's methods.
Runs can use dataset DATASET at `./train_datasets/`. See how dataset is generated at `./scripts/generate_flux_dataset.py`
The run should checkpoint it's weights to `./trained_models/run_<ID>/`. Several checkpoints and artifacts can be saved, e.g. EMA weights and current weights.
Benchmark script would load the model again. So it should also load weights if they already exist after training.

2. Training round

Round is a set of independent runs. Runs can be executed in parallel or sequentially depending on infrastructure.
Each round should contain up to N runs.

3. Direcory of run files: `./src/stdev/upscalers/`

Files from the previous runs are read-only, don't edit them.

4. Infrastructure

This is your main automation tool. It includes the benchmark, run scheduling, and training logs accumulation. It also provides training dataset. You generally should not modify it.

5. Small experiments

These are experiments that **do not require training**.
This includes debugging, micro-optimization tests, inference tweaks.
You can do these experiments **between rounds** while you are thinking on the next training run.
Keep all files related to small experiments in `./dev`

6. Knowledge accumulation

There is a file `./autoresearch/experiment_logs.md` where you can put any notes during experiments.
This file is mostly for you. Treat it as long-term memory. Session context can be reset after some rounds. Experiment logs will not.

Recommended structure:

```
Describe key foundings from previous run

Separate round, e.g. ## Round 3

Describe your hypotheses, ideas, small experiments and ect you made during thinking on the next set of rounds.

Make a small table of runs you made, e.g.

|Run ID              |Run name         |Parent runs     |Run purpose                                |
|--------------------|-----------------|----------------|-------------------------------------------|
|0025                |Larger hidden dim|0005, 0024      |See, if more parameters improve performance|

Exploration graph is usually DAG, so tracking it explicitly is useful.
```

## Experimentation

Each training run has fixed time budget of **30 minutes** (wall clock training time, excluding startup/compilation).

Training code must measure 30 minutes **by itself** including only active training time. Additionaly, infrastructure has time limit guard and would interrupt the run after **35 minutes** if it wouldn't exit by itself.

**What you CAN do:**

- Create new run files in `./src/stdev/upscalers` and modify them. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, inference and sampling strategy, etc.
- Create and modify files and in `./autoresearch`.
- Create and run small code experiments (not full training runs). All runnable files, temporary artifacts should be placed at `./dev`.

**What you CANNOT do:**

- Modify benchmark, evaluation logic, and infrastructure code.
- Modify run files in `./src/stdev/upscalers` from previous rounds. They are read-only archive.
- Install new packages or add dependencies. You can only use what's already in pyproject.toml.

VRAM is a soft constraint. Training run can use all VRAM of the GPU. But at 99% it might be unstable, so leave small gap.

There is no simple goal here: you are jointly optimizing a set of objectives.

They can approximately be ordered by the priority:

1. Benchmark quality metrics (FID and RQI are prioritized)
2. Benchmark computational efficency metrics (Latency and peak VRAM are prioritized)
3. Training loss (for equivalent loss and data setups, for architecture design only)
4. Implementation simplicity (as a term of regularization)

Simplicity criterion: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.


## Single iteration in details

1. Read the benchmark results of previous round.

Benchmark results are in `./raw_results/<dataset_name>/run_<run_ID>/metrics.json`
Profiling results (CPU and CUDA time) is in `./raw_results/<dataset_name>/run_<run_ID>/profiles/<resolution>/`

2. Read `./autoresearch/experiment_logs.md` if you havn't the full context of previous round.

3. Put forward hypotheses, run small experiments if you want, write next N runs.

4. Update `./autoresearch/experiment_logs.md`.

5. Check run (fits in VRAM, no errors) by calling runner with a list of run IDs, e.g.

```bash
./infra/start_round.sh "0002" "0003" "0004" --smoke
```

6. Call execution pipeline with a list of run IDs, e.g.

```bash
./infra/start_round.sh "0002" "0003" "0004"
```

Call this line in the background terminal and set system notification on it's readiness.

## Research Principles

These ones are not strict instructions, but heuristic advice.

### 1. Preserve the best solutions

Maintain the best validated solution found so far. When optimizing multiple objectives, maintain a compact Pareto archive of non-dominated solutions representing meaningful trade-offs.

### 2. Maintain a portfolio of research directions

Track a small number of distinct, promising hypotheses about how to improve the model. Examples include architecture, training dynamics, data usage, optimization, inference algorithms, kernels, and memory or latency trade-offs.

Preserve experiment lineage and record which changes produced which effects. Allow branches to split, combine, or be retired as evidence accumulates.

Prefer approaches with a credible mechanism for improvement, meaningful upside, or high decision value.

### 3. Learn from every round

Analyze results across all runs, not just the winner.

Distinguish:
- Genuine improvements from measurement noise.
- Causal evidence from correlations or confounded changes.
- Failed ideas from inconclusive experiments.
- Local improvements from changes that reveal a more promising research direction.
- Benchmark gains from regressions in other important metrics.

Record unsuccessful experiments and their lessons to avoid repeating them without a specific reason.

### 4. Allocate experiments deliberately

Choose the next experiments according to their expected research value, considering:

- **Expected improvement** to the best solution or Pareto frontier.
- **Information** gained that could change future decisions.
- **Exploration** of promising, materially different approaches.


Balance exploitation of successful approaches with targeted diagnostic experiments and selective exploration. Adjust the balance based on observed progress and remaining uncertainty.
