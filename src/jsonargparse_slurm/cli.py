"""Main CLI entrypoint for jsap-slurm: dispatches containerized jobs to SLURM."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
        dry_run: If True, print the full srun command to stdout and exit.

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
            if arg == f"--{key}" or arg.startswith(f"--{key}.") or arg.startswith(f"--{key}="):
                is_wrapper_arg = True
                break
        if arg in ("--config", "-c", "--print_config"):
            is_wrapper_arg = True

        if is_wrapper_arg:
            wrapper_args.append(arg)
            if "=" not in arg and i + 1 < len(cli_args):
                next_arg = cli_args[i + 1]
                # --config/-c always consume the next token as their value,
                # even if it starts with "-" (e.g. "--config -" = read from stdin).
                # Other wrapper args skip values that look like flags.
                if arg in ("--config", "-c") or not next_arg.startswith("-"):
                    wrapper_args.append(next_arg)
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


def _build_srun_args(
    deploy_cfg: DeploymentConfig,
) -> List[str]:
    """Build the srun command arguments including all SLURM resource flags.

    Args:
        deploy_cfg: The parsed deployment configuration.

    Returns:
        List of srun command arguments.
    """
    args = [
        "srun",
        "--overlap", "-K",
        "--job-name", deploy_cfg.slurm.job_name,
        "--partition", deploy_cfg.slurm.partition,
        "--time", deploy_cfg.slurm.time,
        "--nodes", str(deploy_cfg.slurm.nodes),
        "--ntasks", str(deploy_cfg.slurm.ntasks),
        "--gpus-per-task", str(deploy_cfg.slurm.gpus_per_task),
        "--cpus-per-task", str(deploy_cfg.slurm.cpus_per_task),
        "--mem", deploy_cfg.slurm.mem,
        "--gpu-bind", deploy_cfg.slurm.gpu_bind,
        "--container-image", deploy_cfg.container.image,
    ]
    if deploy_cfg.slurm.mail_user:
        args += ["--mail-user", deploy_cfg.slurm.mail_user]
    if deploy_cfg.slurm.mail_type:
        args += ["--mail-type", deploy_cfg.slurm.mail_type]
    return args


def _expand_mounts(mounts: List[str]) -> str:
    """Expand environment variables like ``$HOME`` in mount paths.

    Used for local execution where the shell won't expand variables.
    """
    return ",".join(os.path.expandvars(m) for m in mounts)


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
        host_netrc_path = container_cfg.netrc_host_path
        if check_exists:
            expanded = Path(host_netrc_path).expanduser()
            if not expanded.exists():
                print(
                    f"WARNING: mount_netrc is True, but {expanded} "
                    "not found on host."
                )
                return mounts
        if host_netrc_path.startswith("~/"):
            host_netrc_path = "$HOME/" + host_netrc_path[2:]
        mounts.append(
            f"{host_netrc_path}:{container_cfg.netrc_container_path}"
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
    """Run the target script directly via ``srun`` (no sbatch).

    If ``deploy_cfg.slurm.ssh_remote`` is set, the srun command is launched
    via SSH as a background job so the SSH connection is not blocked.

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
    container_env = {
        k: v for k, v in deploy_cfg.container.env.items()
        if k != "NVIDIA_DRIVER_CAPABILITIES"
    }
    env_exports = _build_env_exports(container_env)
    target_cmd_str = shlex.join(target_args)

    container_script = _build_container_script(
        clean_yaml_str, repo_setup, target_cmd_str,
        workspace=deploy_cfg.container.workspace,
        run_workspace=deploy_cfg.container.run_workspace,
    )
    if env_exports:
        container_script = f"{env_exports}\n{container_script}"

    srun_args = _build_srun_args(deploy_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = deploy_cfg.slurm.output_path or "."
    log_file = f"{log_dir}/{deploy_cfg.slurm.job_name}_{timestamp}.log"

    def _build_mounts_arg(expand: bool) -> str:
        if expand:
            return f"--container-mounts={_expand_mounts(mounts)}"
        return f"--container-mounts={mounts_str}"

    if deploy_cfg.dry_run:
        base = " ".join(shlex.quote(a) for a in srun_args)
        mounts_part = _build_mounts_arg(expand=not deploy_cfg.slurm.ssh_remote)
        script_part = " ".join(shlex.quote(a) for a in ["/bin/bash", "-c", container_script])
        print(" ".join([
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            base,
            f"--output={shlex.quote(log_file)}",
            mounts_part,
            script_part,
        ]))
        return 0

    if deploy_cfg.slurm.ssh_remote:
        base = " ".join(shlex.quote(a) for a in srun_args)
        remote_srun = (
            f"{base}"
            f" --output {shlex.quote(log_file)}"
            f" --container-mounts={mounts_str}"
            f" {shlex.quote('/bin/bash')} {shlex.quote('-c')} {shlex.quote(container_script)}"
        )
        status_file = f"/tmp/jsap_{timestamp}.status"
        if deploy_cfg.slurm.tmux_session:
            session_name = f"{deploy_cfg.slurm.job_name}_{timestamp}"
            inner_cmd = (
                f"NVIDIA_DRIVER_CAPABILITIES=compute,utility "
                f"mkdir -p {shlex.quote(log_dir)} && "
                f"{remote_srun} </dev/null "
                f"2>{shlex.quote(status_file)}"
            )
            remote_cmd = (
                f'setsid -f tmux new-session -d -s {session_name} "{inner_cmd}"; '
                f'while [ ! -s {shlex.quote(status_file)} ]; do sleep 0.5; done; '
                f"grep 'job [0-9]' {shlex.quote(status_file)} | head -1; "
                f'rm -f {shlex.quote(status_file)}'
            )
            print(f"Submitting via SSH to {deploy_cfg.slurm.ssh_remote}...")
            print(f"tmux session: {session_name}")
        else:
            remote_cmd = (
                "NVIDIA_DRIVER_CAPABILITIES=compute,utility " +
                f"mkdir -p {shlex.quote(log_dir)} && " +
                "setsid -f " + remote_srun +
                f" </dev/null >/dev/null 2>{shlex.quote(status_file)}; " +
                f"while [ ! -s {shlex.quote(status_file)} ]; do sleep 0.5; done; " +
                f"grep 'job [0-9]' {shlex.quote(status_file)} | head -1; " +
                f"rm -f {shlex.quote(status_file)}"
            )
            print(f"Submitting via SSH to {deploy_cfg.slurm.ssh_remote}...")
        ssh_cmd = ["ssh", "-t", deploy_cfg.slurm.ssh_remote, remote_cmd]
        subprocess.run(ssh_cmd, check=True)
        return 0

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    srun_cmd = (
        srun_args
        + ["--output", log_file]
        + ["--container-mounts", _expand_mounts(mounts)]
        + ["/bin/bash", "-c", container_script]
    )
    print(f"Launching srun job: {deploy_cfg.slurm.job_name}")
    env = os.environ.copy()
    env["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility"
    subprocess.Popen(
        srun_cmd,
        start_new_session=True,
        env=env,
    )
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
