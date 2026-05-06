# EvoMem: Memory-Augmented Evolution for Code Optimization

[Python 3.12+](https://www.python.org/downloads/)
[License: MIT](https://opensource.org/licenses/MIT)
[Ruff](https://github.com/astral-sh/ruff)

Anonymous review artifact for a paper on memory mechanisms in LLM-guided
evolutionary program search. GigaEvo uses Large Language Models to automatically
improve programs through iterative mutation and selection (MAP-Elites). Programs
are Python functions; fitness is task performance.

The main contribution in this artifact is the memory mechanism: an ideas tracker
can write deduplicated memory cards from one run, and later evolution runs can
read those cards to condition future mutations.

## Demo

Demo

## Getting Started

- **[Quick Start](docs/QUICKSTART.md)** — Get running in 5 minutes
- **[Architecture Guide](docs/ARCHITECTURE.md)** — System design overview

## Documentation


| Guide                                                | Description                                            |
| ---------------------------------------------------- | ------------------------------------------------------ |
| [DAG System](docs/DAG_SYSTEM.md)                     | Execution engine: stages, dependencies, caching        |
| [Evolution Strategies](docs/EVOLUTION_STRATEGIES.md) | MAP-Elites, multi-island, migration                    |
| [Memory Run Guide](README_memory.md)                 | Two-run workflow for writing and reusing memory cards  |
| [Prompt Co-Evolution](docs/COEVOLUTION.md)           | Co-evolve mutation prompts alongside programs          |
| [Tools](tools/README.md)                             | Analysis, debugging, and problem scaffolding utilities |
| [Usage Guide](docs/USAGE.md)                         | Detailed usage and Hydra configuration                 |
| [Contributing](docs/CONTRIBUTING.md)                 | Guidelines for contributors                            |
| [Changelog](CHANGELOG.md)                            | Version history                                        |


## Quick Start

### 1. Install

**Requirements:** Python 3.12+, Redis

```bash
pip install -e .
```

Install Redis if not already available:

```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Or run via Docker
docker run -d -p 6379:6379 redis:7-alpine
```

### 2. Configure LLM Access

Create a `.env` file with your API key:

```bash
OPENAI_API_KEY=sk-or-v1-your-api-key-here

# Optional: Langfuse tracing
LANGFUSE_PUBLIC_KEY=<key>
LANGFUSE_SECRET_KEY=<key>
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Start Redis

```bash
redis-server
```

### 4. Run Evolution

```bash
python run.py problem.name=heilbron
```

Evolution starts immediately. Logs are saved to `outputs/`.

## How It Works

1. **Load initial programs** from `problems/<name>/initial_programs/`
2. **Mutate programs** using LLMs (GPT, Claude, Gemini, Qwen, etc.),
  optionally conditioned on memory cards
3. **Evaluate fitness** by running each program's `entrypoint()` + `validate()`
4. **Select solutions** using MAP-Elites across a behavior space
5. **Track ideas** and optionally write deduplicated memory cards
6. **Repeat** for N generations

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Problem   │────▶│  Evolution  │────▶│     LLM     │
│  (programs, │     │   Engine    │     │  (mutation)  │
│   metrics)  │     │ (MAP-Elites)│     └──────┬──────┘
└─────────────┘     └──────┬──────┘            │
                           │                   ▼
                    ┌──────┴──────┐     ┌─────────────┐
                    │   Storage   │◀────│  Evaluator   │
                    │   (Redis)   │     │ (DAG Runner) │
                    └─────────────┘     └─────────────┘
```

## Memory Mechanism

The memory workflow is designed as two runs that share a checkpoint folder:

1. Run without memory, but with the ideas tracker enabled, to write memory cards.
2. Run with memory enabled, using the same checkpoint folder as the memory source.

Before building memory cards, ensure these options are enabled in
`config/memory.yaml`:

```yaml
ideas_tracker:
  memory_write_pipeline:
    enabled: true

card_update_dedup:
  enabled: true
```

### Build Memory Cards

```bash
python run.py \
  problem.name=heilbron \
  ideas_tracker=true \
  checkpoint_dir=outputs/memory_bank_01
```

The ideas-tracker run folder will include `memory_write_stats.json` with per-run
write statistics, including `updated` and `rejected` counts.

### Run With Memory

```bash
python run.py \
  problem.name=heilbron \
  memory_enabled=true \
  checkpoint_dir=outputs/memory_bank_01
```

When `memory_enabled=true`, `checkpoint_dir` becomes the memory GAM backend's
`paths.checkpoint_dir` for reading and updating checkpointed memory state. When
`ideas_tracker=true` and `ideas_tracker.memory_write_pipeline.enabled=true`, the
same `checkpoint_dir` is used by the ideas tracker's final write step to store
cards through the memory DB pipeline.

## Customization

### Experiment Presets

```bash
# Steady-state: continuous mutation/evaluation, ~8x throughput
python run.py experiment=steady_state problem.name=heilbron

# Migration bus: parallel runs share rejected programs via Redis stream
python run.py experiment=migration_bus problem.name=heilbron redis.db=0
python run.py experiment=migration_bus problem.name=heilbron redis.db=1

# Steady-state + bus: maximum throughput with cross-run sharing
python run.py experiment=steady_state_bus problem.name=heilbron redis.db=0

# Multi-island evolution (fitness + simplicity islands)
python run.py experiment=multi_island_complexity problem.name=heilbron

# Multi-LLM exploration (diverse mutation models)
python run.py experiment=multi_llm_exploration problem.name=heilbron

# Prompt co-evolution (evolve mutation prompts alongside programs)
python run.py experiment=prompt_coevolution problem.name=heilbron \
    redis.db=4 prompt_fetcher.prompt_redis_db=6
```

### Common Overrides

```bash
# Limit generations
python run.py problem.name=heilbron max_generations=10

# Use different Redis database
python run.py problem.name=heilbron redis.db=5

# Change LLM model
python run.py problem.name=heilbron model_name=anthropic/claude-3.5-sonnet

# Preview config without running
python run.py problem.name=heilbron --cfg job
```

### Prompt Co-Evolution

Co-evolve the mutation prompts alongside your programs. A paired prompt run
evolves the system prompt used by the mutation LLM, selecting for prompts that
produce better mutations:

```bash
# Main run — uses co-evolved prompts from DB 6
python run.py problem.name=my_task pipeline=my_pipeline \
    prompt_fetcher=coevolved prompt_fetcher.prompt_redis_db=6 redis.db=4

# Prompt run — evolves mutation prompts, reads outcomes from DB 4
python run.py problem.name=prompt_evolution pipeline=prompt_evolution \
    redis.db=6 main_redis_db=4 main_redis_prefix=my_task
```

See [Prompt Co-Evolution Guide](docs/COEVOLUTION.md) for the full architecture,
launch instructions, and monitoring.

## Configuration

GigaEvo uses [Hydra](https://hydra.cc/) for modular configuration. All config
files are in `config/`:


| Directory         | Purpose                       | Key files                                                                                                  |
| ----------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `experiment/`     | Complete experiment templates | `base.yaml`, `steady_state.yaml`, `migration_bus.yaml`, `prompt_coevolution.yaml`, `steady_state_bus.yaml` |
| `algorithm/`      | Evolution algorithms          | `single_island.yaml`, `multi_island.yaml`                                                                  |
| `llm/`            | LLM setups                    | `single.yaml`, `heterogeneous.yaml`                                                                        |
| `pipeline/`       | DAG execution pipelines       | `standard.yaml`, `with_context.yaml`, `prompt_evolution.yaml`                                              |
| `prompt_fetcher/` | Prompt sourcing               | `fixed.yaml`, `coevolved.yaml`                                                                             |
| `constants/`      | Tunable parameters            | `evolution.yaml`, `llm.yaml`, `islands.yaml`, `pipeline.yaml`                                              |
| `loader/`         | Program loading               | `directory.yaml`, `redis_selection.yaml`                                                                   |
| `logging/`        | Backends                      | `tensorboard.yaml`, `wandb.yaml`                                                                           |


Override any setting via command line:

```bash
python run.py experiment=full_featured max_generations=50 temperature=0.8
```

## Creating a Problem

1. Create a directory under `problems/`:
  ```
   problems/my_problem/
   ├── validate.py           # Fitness evaluation
   ├── metrics.yaml          # Metric specifications
   ├── task_description.txt  # Problem description for the LLM
   └── initial_programs/     # Seed programs
       ├── strategy1.py      # Must define entrypoint()
       └── strategy2.py
  ```
2. Run:
  ```bash
   python run.py problem.name=my_problem
  ```

Or use the wizard: `python -m tools.wizard config.yaml`

See `problems/heilbron/` for a complete example.

## Output

Results are saved to `outputs/YYYY-MM-DD/HH-MM-SS/`:

- **Logs**: `evolution_*.log`
- **Programs**: Stored in Redis (export with `tools/redis2pd.py`)
- **Metrics**: TensorBoard / W&B (if configured)

## Tools


| Tool                              | Purpose                                      |
| --------------------------------- | -------------------------------------------- |
| `tools/redis2pd.py`               | Export evolution data to CSV/DataFrame       |
| `tools/comparison.py`             | Compare runs with fitness curve plots        |
| `tools/top_programs.py`           | Extract best programs from archive           |
| `tools/flush.py`                  | Safely flush Redis DBs (kills workers first) |
| `tools/experiment/archive_run.sh` | Archive run data before flush                |
| `tools/dag_builder/`              | Visual DAG pipeline designer                 |
| `tools/wizard/`                   | Interactive problem scaffolding              |


See [tools/README.md](tools/README.md) for full documentation and Redis key schema.

## Testing

```bash
# Full test suite (uses fakeredis, no Redis server needed)
python -m pytest

# Specific area
python -m pytest tests/stages/
python -m pytest tests/evolution/

# With coverage
python -m pytest --cov=gigaevo --cov-report=term-missing

# Linting
ruff check . && ruff format --check .
```

## Troubleshooting

**Redis database not empty:**

```bash
# Use tools/flush.py (kills exec_runner workers first):
PYTHONPATH=. python tools/flush.py --db 0 --confirm

# Or use a different DB:
python run.py redis.db=1
```

**LLM connection issues:**

```bash
# Verify API key
echo $OPENAI_API_KEY

# Test OpenRouter
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://openrouter.ai/api/v1/models
```

## License

MIT License — see [LICENSE](LICENSE).

## Citation

Citation information is withheld for anonymous review.