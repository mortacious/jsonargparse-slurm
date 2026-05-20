# jsonargparse-slurm

Deploy Python scripts that use `jsonargparse` (or tools built on it) to a SLURM cluster with a single, unified config file — no custom batch scripts required.

## Motivation

Deploying scripts on SLURM typically means writing one-off batch scripts that hardcode `#SBATCH` directives, container image names, and environment variables. These scripts are fragile, grow stale, and multiply for every experiment variant.

**jsonargparse-slurm** replaces this pattern. You define your SLURM resources, container environment, and git repositories directly in the same YAML config your script already uses. The wrapper handles `sbatch` generation, `srun` invocation, Heredoc-based config injection, and `.netrc` mounting — all transparently. Your script never knows it's running on a cluster.

## Installation

```bash
pip install jsonargparse-slurm
```

Requires Python ≥ 3.10, `jsonargparse >= 4.0`, and `pyyaml >= 6.0`.

## How it works

| Step | What happens |
|------|-------------|
| 1. Parse CLI | `split_by_signature` inspects the parser and dynamically separates wrapper args from target args — no hardcoded prefixes |
| 2. Read YAML | Top-level keys matching `slurm`, `container`, or `repos` are extracted as deployment config; everything else (`training`, `data`, …) is cleaned YAML for the target |
| 3. Parse & validate | Deployment config is parsed by `jsonargparse` with full type checking and defaults |
| 4. Generate SBATCH | SBATCH directives, `srun` container command, repo cloning scripts, and env exports are assembled |
| 5. Submit or dry-run | Submits to SLURM via `sbatch`, or runs locally when `--print_config` is passed |

The cleaned YAML is injected into the container via a Heredoc — it never touches the host filesystem beyond the temp file used for the local dry-run.

## Usage

### Basic: YAML config with defaults

Create a unified config with wrapper keys at the top level alongside your target keys:

```yaml
# config.yaml
slurm:
  job_name: train_job
  partition: gpu
  time: "0-08:00:00"
  nodes: 1
  gpus_per_task: 2
  cpus_per_task: 32
  mem: "128G"

container:
  image: nvcr.io/nvidia/pytorch:23.10-py3
  mount_netrc: false
  env:
    WANDB_API_KEY: your_key_here

training:
  lr: 0.001
  epochs: 100

data:
  path: /data/dataset
```

Then dispatch:

```bash
jsap-slurm --config config.yaml python train.py
```

The target script receives only `training` and `data`. The SLURM and container keys are stripped.

### CLI overrides

Override any deployment value from the command line — takes precedence over YAML:

```bash
jsap-slurm --config config.yaml --slurm.job_name experiment_2 --slurm.time "1-00:00:00" python train.py
```

### No config file — all defaults

If nothing is passed, every parameter falls back to its dataclass default:

```bash
jsap-slurm python train.py
```

### CLI-only overrides (no YAML)

Specify just the values you need:

```bash
jsap-slurm --slurm.job_name quick_test --slurm.partition debug python train.py
```

### Dry run: `--print_config`

Runs the target script locally without submitting to SLURM. Prints the parsed deployment configuration followed by the target's own config dump:

```bash
jsap-slurm --config config.yaml --print_config python train.py
```

The cleaned YAML is written to a temporary file, fed to the target, and deleted afterwards.

### Repositories

Clone and pip-install git repositories inside the container:

```yaml
repos:
  main:
    url: https://github.com/user/project.git
    branch: main
    pip_install: true
  lib:
    url: https://github.com/user/library.git
    branch: dev
    commit: abc1234
    pip_install: false
```

### .netrc support

By default, the host's `~/.netrc` is mounted into the container at `/root/.netrc` so `git clone` can authenticate without hardcoded tokens. Disable it or customize the paths:

```yaml
container:
  mount_netrc: true
  netrc_host_path: ~/.netrc
  netrc_container_path: /root/.netrc
```

## Configuration reference

### SlurmConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `job_name` | `str` | `"jsap_job"` | SLURM job name |
| `partition` | `str` | `"batch"` | SLURM partition |
| `time` | `str` | `"0-04:00:00"` | Wall time (D-HH:MM:SS) |
| `nodes` | `int` | `1` | Number of nodes |
| `ntasks` | `int` | `1` | Number of tasks |
| `gpus_per_task` | `int` | `1` | GPUs per task |
| `cpus_per_task` | `int` | `16` | CPUs per task |
| `mem` | `str` | `"64G"` | Memory per node |
| `gpu_bind` | `str` | `"none"` | GPU binding strategy |

### ContainerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | `str` | `"ubuntu:22.04"` | Container image |
| `mounts` | `List[str]` | `[]` | Additional bind mounts (`host:container`) |
| `env` | `Dict[str, str]` | `{}` | Environment variables exported in container |
| `mount_netrc` | `bool` | `True` | Mount `~/.netrc` into the container |
| `netrc_host_path` | `str` | `"~/.netrc"` | Host-side path to `.netrc` |
| `netrc_container_path` | `str` | `"/root/.netrc"` | Container-side path for `.netrc` |

### RepoConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | `str` | `""` | Git repository URL |
| `branch` | `str` | `"main"` | Branch to clone |
| `commit` | `str` | `"HEAD"` | Specific commit to checkout |
| `pip_install` | `bool` | `True` | Run `pip install --no-cache-dir .` after cloning |
