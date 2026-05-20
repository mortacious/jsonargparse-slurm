"""Main CLI entrypoint for jsap-slurm: dispatches containerized jobs to SLURM."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from .config_schema import (
    ContainerConfig,
    DeploymentConfig,
    RepoConfig,
    SlurmConfig,
)


def _split_args(argv: List[str]) -> Tuple[List[str], List[str]]:
    """Separate wrapper-level args from the target command.

    Args:
        argv: Full command-line arguments starting with the program name.

    Returns:
        Tuple of (wrapper_args, target_args).
        wrapper_args includes ``--config``, ``--print_config`` and any
        ``--deployment.*`` overrides. Everything else is the target command.
    """
    wrapper_args: List[str] = []
    rest: List[str] = []
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("--config", "-c"):
            wrapper_args.append(arg)
            if i + 1 < len(argv):
                wrapper_args.append(argv[i + 1])
                i += 2
                continue
        elif arg == "--print_config":
            wrapper_args.append(arg)
        elif arg.startswith("--deployment.") or arg.startswith("--deployment="):
            wrapper_args.append(arg)
            if "=" not in arg and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                wrapper_args.append(argv[i + 1])
                i += 2
                continue
        else:
            rest.append(arg)
        i += 1
    return wrapper_args, rest


def _parse_wrapper_args(
    wrapper_args: List[str],
) -> Tuple[Optional[str], bool, dict]:
    """Parse wrapper-level arguments to extract config path, print flag, and deployment overrides.

    Args:
        wrapper_args: The wrapper-level arguments extracted by :func:`_split_args`.

    Returns:
        Tuple of (config_path, is_print_config, deployment_overrides dict).
        deployment_overrides maps dotted keys (e.g. ``slurm.job_name``) to values.
    """
    config_path: Optional[str] = None
    is_print_config = False
    deployment_overrides: dict = {}

    i = 0
    while i < len(wrapper_args):
        arg = wrapper_args[i]
        if arg in ("--config", "-c") and i + 1 < len(wrapper_args):
            config_path = wrapper_args[i + 1]
            i += 2
            continue
        elif arg == "--print_config":
            is_print_config = True
        elif arg.startswith("--deployment."):
            key = arg[len("--deployment.") :]
            if "=" in key:
                key, value = key.split("=", 1)
                deployment_overrides[key] = value
            else:
                i += 2
                value = wrapper_args[i - 1]
                deployment_overrides[key] = value
                continue
        elif arg.startswith("--deployment="):
            key, value = arg[len("--deployment=") :].split("=", 1)
            deployment_overrides[key] = value
        i += 1

    return config_path, is_print_config, deployment_overrides


def _apply_deployment_overrides(deploy_cfg: DeploymentConfig, overrides: dict) -> None:
    """Apply CLI deployment overrides to a DeploymentConfig instance in-place.

    Supports dotted keys such as ``slurm.job_name``, ``container.image``,
    and nested dicts like ``repos.main.url``.

    Args:
        deploy_cfg: The DeploymentConfig instance to modify.
        overrides: Dict of dotted key -> string value pairs.
    """
    for key_path, value_str in overrides.items():
        parts = key_path.split(".")
        target: object = deploy_cfg
        for part in parts[:-1]:
            if isinstance(target, dict):
                target = target[part]
            else:
                target = getattr(target, part)
        field = parts[-1]
        if isinstance(target, dict):
            current = target.get(field)
        else:
            current = getattr(target, field)
        if current is not None:
            if isinstance(current, bool):
                value = value_str.lower() in ("true", "1", "yes")
            else:
                try:
                    value = type(current)(value_str)
                except (ValueError, TypeError):
                    value = value_str
        else:
            value = value_str
        if isinstance(target, dict):
            target[field] = value
        else:
            setattr(target, field, value)


def _build_deployment_config(yaml_deployment: dict) -> DeploymentConfig:
    """Construct a DeploymentConfig from a YAML deployment dict.

    Args:
        yaml_deployment: The ``deployment`` section from the YAML config.

    Returns:
        Fully populated :class:`DeploymentConfig`.
    """
    slurm_data = yaml_deployment.get("slurm", {})
    slurm = SlurmConfig(**{
        k: v for k, v in slurm_data.items()
        if k in SlurmConfig.__dataclass_fields__
    })

    container_data = yaml_deployment.get("container", {})
    container = ContainerConfig(**{
        k: v for k, v in container_data.items()
        if k in ContainerConfig.__dataclass_fields__
    })

    repos_data = yaml_deployment.get("repos", {})
    repos = {}
    for name, repo_data in repos_data.items():
        repos[name] = RepoConfig(**{
            k: v for k, v in repo_data.items()
            if k in RepoConfig.__dataclass_fields__
        })

    return DeploymentConfig(slurm=slurm, container=container, repos=repos)


def _extract_config_path(args: List[str]) -> Optional[Path]:
    """Extract the config file path from the given argument list.

    Looks for ``--config`` or ``-c`` followed by a path value.

    Args:
        args: List of command-line arguments to scan.

    Returns:
        Resolved ``Path`` to the config file, or ``None`` if not found.
    """
    for i, arg in enumerate(args):
        if arg in ("--config", "-c") and i + 1 < len(args):
            return Path(args[i + 1]).resolve()
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1]).resolve()
        if arg.startswith("-c="):
            return Path(arg[3:]).resolve()
    return None


def _has_print_config(args: List[str]) -> bool:
    """Check whether ``--print_config`` appears in the argument list.

    Args:
        args: List of command-line arguments.

    Returns:
        ``True`` if ``--print_config`` is present.
    """
    return "--print_config" in args


def _build_repo_setup_script(repos: dict) -> str:
    """Generate shell commands to clone and optionally install repositories.

    Args:
        repos: Mapping of repo name to :class:`RepoConfig` instances.

    Returns:
        Multi-line bash snippet for cloning and setting up repositories.
    """
    lines: List[str] = []
    for name, info in repos.items():
        lines.append(f"echo '-> Cloning {name}...'")
        lines.append(f"git clone --branch {info.branch} {info.url} {name}")
        lines.append(f"cd {name}")
        if info.commit != "HEAD":
            lines.append(f"git checkout {info.commit}")
        if info.pip_install:
            lines.append("pip install --no-cache-dir .")
        lines.append("cd ..")
    return "\n".join(lines)


def _build_env_exports(env: dict) -> str:
    """Generate shell ``export`` statements from an environment dict.

    Args:
        env: Mapping of environment variable names to values.

    Returns:
        Multi-line bash snippet setting environment variables.
    """
    return "\n".join(f'export {k}="{v}"' for k, v in env.items())


def _build_container_script(
    clean_yaml_str: str,
    repo_setup: str,
    target_cmd_str: str,
) -> str:
    """Assemble the inline container bash script with Heredoc config injection.

    Args:
        clean_yaml_str: The cleaned YAML content (without the deployment block).
        repo_setup: Shell commands for repository cloning and setup.
        target_cmd_str: Shell-escaped target command string.

    Returns:
        Complete bash script string to run inside the container via srun.
    """
    return f"""set -e
export PATH="/opt/conda/envs/perception_env/bin:$PATH"

WORKSPACE="/tmp/custom_code"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

cat << 'EOF_CONFIG' > /tmp/clean_config_$$.yaml
{clean_yaml_str}
EOF_CONFIG

{repo_setup}

echo "=== Executing Target Script ==="
{target_cmd_str} --config /tmp/clean_config_$$.yaml

rm /tmp/clean_config_$$.yaml
"""


def _build_sbatch_content(
    deploy_cfg: DeploymentConfig,
    env_exports: str,
    container_script: str,
    mounts_str: str,
    log_dir: Path,
    timestamp: str,
) -> Tuple[str, Path]:
    """Build the SBATCH submission script content and return it with its file path.

    Args:
        deploy_cfg: The parsed deployment configuration.
        env_exports: Shell ``export`` statements for container environment.
        container_script: The inline container bash script.
        mounts_str: Comma-separated container mount specifications.
        log_dir: Directory for SLURM output logs.
        timestamp: Timestamp string for the sbatch filename.

    Returns:
        Tuple of (sbatch script content as string, Path to the sbatch file).
    """
    log_file = log_dir / f"{deploy_cfg.slurm.job_name}_%j.out"
    sbatch_file = log_dir / f"{deploy_cfg.slurm.job_name}_{timestamp}.sbatch"

    content = f"""#!/bin/bash
#SBATCH --job-name={deploy_cfg.slurm.job_name}
#SBATCH --output={log_file}
#SBATCH --partition={deploy_cfg.slurm.partition}
#SBATCH --time={deploy_cfg.slurm.time}
#SBATCH --nodes={deploy_cfg.slurm.nodes}
#SBATCH --ntasks={deploy_cfg.slurm.ntasks}
#SBATCH --gpus-per-task={deploy_cfg.slurm.gpus_per_task}
#SBATCH --cpus-per-task={deploy_cfg.slurm.cpus_per_task}
#SBATCH --mem={deploy_cfg.slurm.mem}
#SBATCH --gpu-bind={deploy_cfg.slurm.gpu_bind}

{env_exports}

CONTAINER_CMD=$(cat << 'EOF'
{container_script}
EOF
)

srun -K \\
     --container-image={deploy_cfg.container.image} \\
     --container-mounts={mounts_str} \\
     /bin/bash -c "$CONTAINER_CMD"
"""
    return content, sbatch_file


def _resolve_mounts(deploy_cfg: DeploymentConfig) -> List[str]:
    """Resolve container mounts including the optional ``.netrc`` mount.

    Args:
        deploy_cfg: The parsed deployment configuration.

    Returns:
        List of mount specifications (``host:container`` strings).
    """
    mounts = list(deploy_cfg.container.mounts)
    if deploy_cfg.container.mount_netrc:
        host_netrc = Path(deploy_cfg.container.netrc_host_path).expanduser().resolve()
        if host_netrc.exists():
            mounts.append(
                f"{host_netrc}:{deploy_cfg.container.netrc_container_path}"
            )
        else:
            print(
                f"WARNING: mount_netrc is True, but {host_netrc} "
                "not found on host."
            )
    return mounts


def _run_print_config_locally(
    target_args: List[str],
    clean_yaml_str: str,
) -> int:
    """Execute the target script locally with the cleaned config via a temp file.

    Args:
        target_args: The target command arguments.
        clean_yaml_str: Cleaned YAML content.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    print("=== Evaluating config locally (--print_config) ===")
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml"
    ) as tmp:
        tmp.write(clean_yaml_str)
        local_tmp_config = tmp.name

    local_cmd = target_args.copy()
    local_cmd.extend(["--config", local_tmp_config, "--print_config"])

    try:
        subprocess.run(local_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Local execution failed: {e}")
        return 1
    finally:
        if os.path.exists(local_tmp_config):
            os.remove(local_tmp_config)
    return 0


def _dispatch_slurm_job(
    target_args: List[str],
    deploy_cfg: DeploymentConfig,
    clean_yaml_str: str,
) -> int:
    """Generate the SBATCH script and submit it to SLURM.

    Args:
        target_args: The target command arguments.
        deploy_cfg: The parsed deployment configuration.
        clean_yaml_str: Cleaned YAML content.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    mounts = _resolve_mounts(deploy_cfg)
    mounts_str = ",".join(mounts)

    repo_setup = _build_repo_setup_script(deploy_cfg.repos)
    env_exports = _build_env_exports(deploy_cfg.container.env)
    target_cmd_str = shlex.join(target_args)

    container_script = _build_container_script(
        clean_yaml_str, repo_setup, target_cmd_str
    )

    log_dir = Path("logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sbatch_content, sbatch_file = _build_sbatch_content(
        deploy_cfg, env_exports, container_script, mounts_str, log_dir, timestamp
    )

    with open(sbatch_file, "w") as f:
        f.write(sbatch_content)

    print(f"Deployment generated: {sbatch_file}")
    subprocess.run(["sbatch", str(sbatch_file)], check=True)
    return 0


def main() -> int:
    """Main entry point for the ``jsap-slurm`` CLI.

    Parses a unified YAML configuration, extracts the ``deployment`` block,
    strips it from the YAML, and either runs the target locally (for
    ``--print_config`` dry runs) or dispatches the job to SLURM.

    Returns:
        Exit code (0 on success, non-zero on error).
    """
    wrapper_args, target_args = _split_args(sys.argv)
    config_path_str, is_print_config, deployment_overrides = _parse_wrapper_args(
        wrapper_args
    )

    if not config_path_str:
        print("ERROR: A config file must be provided via '--config'.")
        return 1
    config_path = Path(config_path_str).resolve()
    if not config_path.exists():
        print("ERROR: Config file not found: {}".format(config_path))
        return 1

    with open(config_path, "r") as f:
        raw_yaml = yaml.safe_load(f)

    yaml_deployment = raw_yaml.pop("deployment", None)
    if yaml_deployment is None:
        print("ERROR: No 'deployment' section found in the provided config.")
        return 1
    deploy_cfg = _build_deployment_config(yaml_deployment)
    _apply_deployment_overrides(deploy_cfg, deployment_overrides)

    clean_yaml_str = yaml.dump(raw_yaml, sort_keys=False)

    if is_print_config:
        return _run_print_config_locally(target_args, clean_yaml_str)

    return _dispatch_slurm_job(target_args, deploy_cfg, clean_yaml_str)


if __name__ == "__main__":
    sys.exit(main())
