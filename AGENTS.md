# AGENTS.md — jsonargparse-slurm

## Quick start

```bash
python3 -m venv /tmp/venv && source /tmp/venv/bin/activate
pip install -e ".[test]"
pytest tests/
```

The CLI entry point is `jsap-slurm` → `jsonargparse_slurm.cli:main`.

## Architecture (must understand before touching parser logic)

The parser is **signature-driven** — no hardcoded argument prefixes.

1. `cli.setup_cluster(slurm, container, repos)` is a function whose parameter names *are* the wrapper argument names. `auto_parser(setup_cluster)` generates a jsonargparse `ArgumentParser` from its signature.
2. `cli.split_by_signature(parser, argv, raw_yaml)` inspects `parser._actions` to find known top-level keys (`slurm`, `container`, `repos`), then splits YAML keys and CLI args into wrapper vs target portions.
3. Wrapper YAML keys are fed via `parser.set_defaults(**wrapper_yaml)` — **never** via `--config` (see below). Target YAML keys become the cleaned YAML string passed to the target script.
4. CLI overrides (e.g. `--slurm.job_name=myjob`) go through the parser normally. Precedence: CLI > YAML > function defaults > dataclass defaults.

## YAML format

Wrapper keys are **top-level**, not nested inside a `deployment:` block:

```yaml
slurm:
  job_name: train_job
  partition: gpu
container:
  image: nvcr.io/nvidia/pytorch:23.10-py3
  mount_netrc: false
training:          # ← target key, forwarded to the script
  lr: 0.001
```

## jsonargparse quirks (4.49.0)

- **`parse_known_args()` raises `NotImplementedError`** — do not use it.
- **`action='store_true'` / `nargs=0` flags do not work** with `add_argument` — jsonargparse always tries to parse a value. This is why `--print_config` and `--config` are filtered out by `_filter_parser_args` before the parser sees them.
- **`auto_parser` auto-adds `--config` with `ActionConfigFile`** — this would fail on unknown YAML keys like `training.lr` if allowed to run. The YAML is loaded manually, split via `split_by_signature`, and wrapper keys are fed via `set_defaults`. `--config` and its value are stripped from args before parsing.
- **Boolean `add_argument` with `type=bool`** requires `--flag=true` syntax. The wrapper avoids this entirely by handling `--print_config` outside the parser.

## Testing

- All SLURM interactions are mocked: patch `jsonargparse_slurm.cli.subprocess.run`, not just `subprocess.run`.
- Tests that exercise dispatch use `os.chdir(tmp_path)` so that `Path("logs").resolve()` creates `logs/` inside the temp dir.
- Verify generated SBATCH content via `tmp_path.glob("logs/*.sbatch")` and reading the file text.
- Use `capsys` fixture when checking stdout output (e.g. `--print_config` prints deployment config).

## Key files

| File | Role |
|------|------|
| `src/jsonargparse_slurm/config_schema.py` | Dataclasses: SlurmConfig, ContainerConfig, RepoConfig, DeploymentConfig |
| `src/jsonargparse_slurm/cli.py` | CLI entrypoint, `setup_cluster`, `split_by_signature`, `_filter_parser_args`, sbatch generation |
| `tests/test_cli.py` | All CLI unit tests (57 tests) |
| `tests/test_config_schema.py` | Dataclass default/value tests (8 tests) |
| `README.md` | User-facing documentation and usage examples |
| `pyproject.toml` | Dependencies: `jsonargparse>=4.0`, `pyyaml>=6.0`; test: `pytest>=8.0` |
