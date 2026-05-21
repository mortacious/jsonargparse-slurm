"""Main CLI entrypoint for jsap-slurm: dispatches containerized jobs to SLURM."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from jsonargparse import auto_parser, ArgumentParser

from .config_schema import (
    ContainerConfig,
    DeploymentConfig,
    RepoConfig,
    SlurmConfig,
)


def setup_cluster(
    slurm: SlurmConfig = SlurmConfig(),
    container: ContainerConfig = ContainerConfig(),
    repos: Dict[str, RepoConfig] = {},
    dry_run: bool = False,
) -> DeploymentConfig:
    """Entry-point function whose signature drives the wrapper parser.

    Args:
        slurm: SLURM resource allocation settings.
        container: Container image and runtime settings.
        repos: Git repositories to clone inside the container.
        dry_run: If True, print the full SBATCH script to stdout and exit.

    Returns:
        Consolidated :class:`DeploymentConfig`.
    """
    return DeploymentConfig(
        slurm=slurm, container=container, repos=repos, dry_run=dry_run,
    )


def split_by_signature(
    parser: ArgumentParser,
    cli_args: List[str],
    raw_yaml: Dict,
) -> Tuple[List[str], List[str], Dict, Dict]:
    """Split CLI arguments and YAML keys by the parser's known top-level parameters.

    Dynamically identifies which top-level keys the parser recognizes (based on
    the ``setup_cluster`` function signature) and separates CLI args and YAML
    keys accordingly.

    ``--config``, ``-c`` and ``--print_config`` are always treated as wrapper
    arguments since they control the wrapper behaviour, not the target script.

    Args:
        parser: The jsonargparse :class:`ArgumentParser` whose signature determines
            which args/keys belong to the wrapper.
        cli_args: Full CLI argument list (without the program name).
        raw_yaml: Parsed YAML dictionary from the config file.

    Returns:
        Tuple of ``(wrapper_args, target_args, wrapper_yaml, target_yaml)``.
    """
    known_keys: set = set()
    for action in parser._actions:
        if action.dest and action.dest not in ("help", "config", "print_config"):
            known_keys.add(action.dest.split(".")[0])

    wrapper_yaml: Dict = {}
    target_yaml: Dict = {}
    for key, value in raw_yaml.items():
        if key in known_keys:
            wrapper_yaml[key] = value
        else:
            target_yaml[key] = value

    wrapper_args: List[str] = []
    target_args: List[str] = []
    i = 0
    while i < len(cli_args):
        arg = cli_args[i]
        is_wrapper_arg = False
        for key in known_keys:
            if arg == f"--{key}" or arg.startswith(f"--{key}."):
                is_wrapper_arg = True
                break
        if arg in ("--config", "-c", "--print_config"):
            is_wrapper_arg = True

        if is_wrapper_arg:
            wrapper_args.append(arg)
            if "=" not in arg and i + 1 < len(cli_args) and not cli_args[i + 1].startswith("-"):
                wrapper_args.append(cli_args[i + 1])
                i += 1
        else:
            target_args.append(arg)
        i += 1

    return wrapper_args, target_args, wrapper_yaml, target_yaml


def _filter_parser_args(wrapper_args: List[str]) -> List[str]:
    """Remove ``--config``, ``-c`` and ``--print_config`` entries from the arg list.

    These are consumed manually by the wrapper and must not be passed to the
    jsonargparse parser again since the YAML has already been fed via
    :meth:`set_defaults`.

    Args:
        wrapper_args: The wrapper-side arguments.

    Returns:
        Filtered argument list suitable for ``parser.parse_args()``.
    """
    result: List[str] = []
    skip = False
    for arg in wrapper_args:
        if skip:
            skip = False
            continue
        if arg in ("--config", "-c"):
            skip = True
            continue
        if arg == "--print_config":
            continue
        result.append(arg)
    return result


def _build_repo_setup_script(repos: dict) -> str:
    """Generate shell commands to clone and optionally install repositories.

    Args:
        repos: Mapping of repo name to :class:`.RepoConfig` instances or dicts.

    Returns:
        Multi-line bash snippet for cloning and setting up repositories.
    """
    lines: List[str] = []
    for name, info in repos.items():
        if isinstance(info, dict):
            info = RepoConfig(**info)
        lines.append(f"echo '-> Cloning {name}...'")
        if info.target_path:
            target = info.target_path
            lines.append(f"mkdir -p \"$(dirname {target})\"")
            lines.append(f"git clone --branch {info.branch} {info.url} {target}")
            lines.append(f"cd {target}")
            if info.commit != "HEAD":
                lines.append(f"git checkout {info.commit}")
            if info.pip_install:
                lines.append("pip install --no-cache-dir .")
            lines.append("cd \"$WORKSPACE\"")
        else:
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
    workspace: str = "/workspace",
    run_workspace: str = "/workspace",
) -> str:
    """Assemble the inline container bash script with Heredoc config injection.

    Args:
        clean_yaml_str: The cleaned YAML content (without wrapper keys).
        repo_setup: Shell commands for repository cloning and setup.
        target_cmd_str: Shell-escaped target command string.
        workspace: Directory where repositories are cloned.
        run_workspace: Directory where the target script is executed.

    Returns:
        Complete bash script string to run inside the container via srun.
    """
    return f"""set -e
export PATH="/opt/conda/envs/perception_env/bin:$PATH"

WORKSPACE="{workspace}"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

cat << 'EOF_CONFIG' > /tmp/clean_config_$$.yaml
{clean_yaml_str}
EOF_CONFIG

{repo_setup}

RUN_WORKSPACE="{run_workspace}"
mkdir -p "$RUN_WORKSPACE"
cd "$RUN_WORKSPACE"

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

    mail_directives = ""
    if deploy_cfg.slurm.mail_user:
        mail_directives += f"#SBATCH --mail-user={deploy_cfg.slurm.mail_user}\n"
    if deploy_cfg.slurm.mail_type:
        mail_directives += f"#SBATCH --mail-type={deploy_cfg.slurm.mail_type}\n"
    mail_directives = mail_directives.rstrip("\n")

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
{mail_directives}
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


def _resolve_mounts(
    container_cfg: ContainerConfig,
    check_exists: bool = True,
) -> List[str]:
    """Resolve container mounts including the optional ``.netrc`` mount.

    Args:
        container_cfg: The parsed container configuration.
        check_exists: Whether to verify that mount source paths exist locally.
            Set to ``False`` when submitting via SSH, since the file lives on
            the remote login node.

    Returns:
        List of mount specifications (``host:container`` strings).
    """
    mounts = list(container_cfg.mounts)
    if container_cfg.mount_netrc:
        host_netrc = Path(container_cfg.netrc_host_path).expanduser().resolve()
        if check_exists and not host_netrc.exists():
            print(
                f"WARNING: mount_netrc is True, but {host_netrc} "
                "not found on host."
            )
        else:
            mounts.append(
                f"{host_netrc}:{container_cfg.netrc_container_path}"
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

    If ``deploy_cfg.slurm.ssh_remote`` is set, pipes the sbatch script to the
    remote host via SSH instead of writing a local file and calling ``sbatch``
    directly.

    Args:
        target_args: The target command arguments.
        deploy_cfg: The parsed deployment configuration.
        clean_yaml_str: Cleaned YAML content.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    mounts = _resolve_mounts(
        deploy_cfg.container,
        check_exists=not deploy_cfg.slurm.ssh_remote,
    )
    mounts_str = ",".join(mounts)

    repo_setup = _build_repo_setup_script(deploy_cfg.repos)
    env_exports = _build_env_exports(deploy_cfg.container.env)
    target_cmd_str = shlex.join(target_args)

    container_script = _build_container_script(
        clean_yaml_str, repo_setup, target_cmd_str,
        workspace=deploy_cfg.container.workspace,
        run_workspace=deploy_cfg.container.run_workspace,
    )

    log_dir = Path("logs").resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sbatch_content, _ = _build_sbatch_content(
        deploy_cfg, env_exports, container_script, mounts_str, log_dir, timestamp
    )
    if deploy_cfg.dry_run:
        # Print the SBATCH script and exit
        print(sbatch_content)
        return 0

    if deploy_cfg.slurm.ssh_remote:
        print(f"Submitting via SSH to {deploy_cfg.slurm.ssh_remote}...")
        ssh_cmd = ["ssh", deploy_cfg.slurm.ssh_remote, "sbatch"]
        subprocess.run(ssh_cmd, input=sbatch_content, text=True, check=True)
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    sbatch_file = log_dir / f"{deploy_cfg.slurm.job_name}_{timestamp}.sbatch"

    with open(sbatch_file, "w") as f:
        f.write(sbatch_content)

    print(f"Deployment generated: {sbatch_file}")
    subprocess.run(["sbatch", str(sbatch_file)], check=True)
    return 0


def main() -> int:
    """Main entry point for the ``jsap-slurm`` CLI.

    Uses ``auto_parser`` to generate a parser from the ``setup_cluster``
    function signature, dynamically splits CLI args and YAML keys into
    wrapper and target portions, and either runs locally (``--print_config``)
    or dispatches to SLURM.

    Returns:
        Exit code (0 on success, non-zero on error).
    """
    parser = auto_parser(
        setup_cluster,
        description="jsap-slurm: JSONArgParse SLURM Wrapper",
    )

    config_path = None
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        config_path = Path(sys.argv[idx + 1]).resolve()
    elif "-c" in sys.argv:
        idx = sys.argv.index("-c")
        config_path = Path(sys.argv[idx + 1]).resolve()

    raw_yaml = {}
    if config_path:
        if not config_path.exists():
            print("ERROR: Config file not found: {}".format(config_path))
            return 1
        with open(config_path, "r") as f:
            raw_yaml = yaml.safe_load(f) or {}

    wrapper_args, target_args, wrapper_yaml, target_yaml = split_by_signature(
        parser, sys.argv[1:], raw_yaml
    )

    parser.set_defaults(**wrapper_yaml)
    filtered_args = _filter_parser_args(wrapper_args)
    cfg = parser.parse_args(filtered_args)
    deploy_cfg = DeploymentConfig(
        slurm=cfg.slurm,
        container=cfg.container,
        repos=getattr(cfg, "repos", {}),
        dry_run=getattr(cfg, "dry_run", False),
    )

    clean_yaml_str = yaml.dump(target_yaml, sort_keys=False)

    is_print_config = (
        "--print_config" in sys.argv
        or (len(sys.argv) == 2 and sys.argv[1] == "--print_config")
    )
    if is_print_config:
        print(parser.dump(cfg, format="yaml"))
        return _run_print_config_locally(target_args, clean_yaml_str)

    return _dispatch_slurm_job(target_args, deploy_cfg, clean_yaml_str)


if __name__ == "__main__":
    sys.exit(main())
