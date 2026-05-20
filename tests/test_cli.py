"""Tests for the CLI module with mocked SLURM interactions."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from jsonargparse_slurm.cli import (
    _split_args,
    _parse_wrapper_args,
    _apply_deployment_overrides,
    _build_deployment_config,
    _extract_config_path,
    _has_print_config,
    _build_repo_setup_script,
    _build_env_exports,
    _build_container_script,
    _resolve_mounts,
    _build_sbatch_content,
    _run_print_config_locally,
    _dispatch_slurm_job,
    main,
)
from jsonargparse_slurm.config_schema import (
    DeploymentConfig,
    RepoConfig,
    SlurmConfig,
    ContainerConfig,
)


def _sample_yaml():
    return {
        "training": {
            "lr": 0.001,
            "epochs": 100,
        },
        "data": {
            "path": "/data/dataset",
        },
        "deployment": {
            "slurm": {
                "job_name": "train_job",
                "partition": "gpu",
                "time": "0-08:00:00",
                "nodes": 1,
                "ntasks": 1,
                "gpus_per_task": 2,
                "cpus_per_task": 32,
                "mem": "128G",
                "gpu_bind": "none",
            },
            "container": {
                "image": "nvcr.io/nvidia/pytorch:23.10-py3",
                "mounts": ["/data:/data"],
                "env": {"WANDB_API_KEY": "test123"},
                "mount_netrc": False,
            },
        },
    }


def _sample_deployment_config():
    return DeploymentConfig(
        slurm=SlurmConfig(
            job_name="train_job",
            partition="gpu",
            time="0-08:00:00",
            nodes=1,
            ntasks=1,
            gpus_per_task=2,
            cpus_per_task=32,
            mem="128G",
            gpu_bind="none",
        ),
        container=ContainerConfig(
            image="nvcr.io/nvidia/pytorch:23.10-py3",
            mounts=["/data:/data"],
            env={"WANDB_API_KEY": "test123"},
            mount_netrc=False,
        ),
    )


class TestSplitArgs:
    """Tests for _split_args."""

    def test_separates_wrapper_from_target(self):
        wrapper, target = _split_args([
            "jsap-slurm",
            "--config", "config.yaml",
            "python", "train.py",
        ])
        assert "--config" in wrapper
        assert "config.yaml" in wrapper
        assert target == ["python", "train.py"]

    def test_handles_print_config(self):
        wrapper, target = _split_args([
            "jsap-slurm",
            "--print_config",
            "--config", "config.yaml",
            "python", "train.py",
            "--integrator.min_points", "10",
        ])
        assert "--print_config" in wrapper
        assert "--config" in wrapper
        assert "python" in target
        assert "--integrator.min_points" in target
        assert "10" in target

    def test_handles_deployment_overrides(self):
        wrapper, target = _split_args([
            "jsap-slurm",
            "--deployment.slurm.job_name", "myjob",
            "--deployment.slurm.partition", "gpu",
            "--config", "config.yaml",
            "python", "train.py",
        ])
        assert "--deployment.slurm.job_name" in wrapper
        assert "myjob" in wrapper
        assert "--deployment.slurm.partition" in wrapper
        assert "gpu" in wrapper
        assert target == ["python", "train.py"]

    def test_handles_deployment_equals_form(self):
        wrapper, target = _split_args([
            "jsap-slurm",
            "--deployment.slurm.job_name=myjob",
            "--config", "config.yaml",
            "python", "train.py",
        ])
        assert "--deployment.slurm.job_name=myjob" in wrapper
        assert target == ["python", "train.py"]

    def test_target_only_args(self):
        wrapper, target = _split_args([
            "jsap-slurm",
            "--config", "config.yaml",
            "python", "train.py",
            "--integrator.min_points", "10",
            "--model.arch", "resnet50",
        ])
        assert "--integrator.min_points" in target
        assert "--model.arch" in target
        assert "10" in target


class TestParseWrapperArgs:
    """Tests for _parse_wrapper_args."""

    def test_extracts_config_and_overrides(self):
        config_path, is_print, overrides = _parse_wrapper_args([
            "--config", "/path/to/config.yaml",
            "--deployment.slurm.job_name", "myjob",
            "--print_config",
        ])
        assert config_path == "/path/to/config.yaml"
        assert is_print is True
        assert overrides == {"slurm.job_name": "myjob"}

    def test_deployment_equals_form(self):
        config_path, is_print, overrides = _parse_wrapper_args([
            "--deployment.slurm.job_name=myjob",
            "--config", "config.yaml",
        ])
        assert config_path == "config.yaml"
        assert overrides == {"slurm.job_name": "myjob"}

    def test_no_deployment_overrides(self):
        config_path, is_print, overrides = _parse_wrapper_args([
            "--config", "config.yaml",
        ])
        assert config_path == "config.yaml"
        assert is_print is False
        assert overrides == {}

    def test_no_print_config_flag(self):
        config_path, is_print, overrides = _parse_wrapper_args([
            "--config", "config.yaml",
        ])
        assert is_print is False


class TestApplyDeploymentOverrides:
    """Tests for _apply_deployment_overrides."""

    def test_overrides_slurm_field(self):
        cfg = DeploymentConfig()
        _apply_deployment_overrides(cfg, {"slurm.job_name": "cli_job"})
        assert cfg.slurm.job_name == "cli_job"

    def test_overrides_container_field(self):
        cfg = DeploymentConfig()
        _apply_deployment_overrides(cfg, {"container.image": "custom:v2"})
        assert cfg.container.image == "custom:v2"

    def test_overrides_integer_field(self):
        cfg = DeploymentConfig()
        _apply_deployment_overrides(cfg, {"slurm.nodes": "4"})
        assert cfg.slurm.nodes == 4

    def test_overrides_bool_field(self):
        cfg = DeploymentConfig()
        _apply_deployment_overrides(cfg, {"container.mount_netrc": "False"})
        assert cfg.container.mount_netrc is False

    def test_multiple_overrides(self):
        cfg = DeploymentConfig()
        _apply_deployment_overrides(cfg, {
            "slurm.job_name": "multi",
            "slurm.partition": "gpu",
            "container.image": "img:v1",
        })
        assert cfg.slurm.job_name == "multi"
        assert cfg.slurm.partition == "gpu"
        assert cfg.container.image == "img:v1"


class TestBuildDeploymentConfig:
    """Tests for _build_deployment_config."""

    def test_constructs_from_yaml_dict(self):
        yaml_dep = {
            "slurm": {
                "job_name": "yaml_job",
                "partition": "debug",
            },
            "container": {
                "image": "img:v1",
            },
        }
        cfg = _build_deployment_config(yaml_dep)
        assert cfg.slurm.job_name == "yaml_job"
        assert cfg.slurm.partition == "debug"
        assert cfg.slurm.nodes == 1
        assert cfg.container.image == "img:v1"
        assert cfg.container.mount_netrc is True

    def test_repos_constructed(self):
        yaml_dep = {
            "repos": {
                "main": {
                    "url": "https://github.com/user/repo.git",
                    "branch": "dev",
                }
            }
        }
        cfg = _build_deployment_config(yaml_dep)
        assert "main" in cfg.repos
        assert cfg.repos["main"].url == "https://github.com/user/repo.git"
        assert cfg.repos["main"].branch == "dev"
        assert cfg.repos["main"].pip_install is True


class TestExtractConfigPath:
    """Tests for _extract_config_path."""

    def test_double_dash_config(self):
        path = _extract_config_path(["python", "train.py", "--config", "/tmp/cfg.yaml"])
        assert path == Path("/tmp/cfg.yaml").resolve()

    def test_short_config_flag(self):
        path = _extract_config_path(["python", "train.py", "-c", "cfg.yaml"])
        assert path == Path("cfg.yaml").resolve()

    def test_equals_form(self):
        path = _extract_config_path(["python", "train.py", "--config=/tmp/cfg.yaml"])
        assert path == Path("/tmp/cfg.yaml").resolve()

    def test_no_config_returns_none(self):
        path = _extract_config_path(["python", "train.py"])
        assert path is None

    def test_config_without_value_returns_none(self):
        path = _extract_config_path(["--config"])
        assert path is None


class TestHasPrintConfig:
    """Tests for _has_print_config."""

    def test_detects_print_config(self):
        assert _has_print_config(["python", "train.py", "--print_config"]) is True

    def test_no_print_config(self):
        assert _has_print_config(["python", "train.py", "--config", "cfg.yaml"]) is False

    def test_empty_args(self):
        assert _has_print_config([]) is False


class TestBuildRepoSetupScript:
    """Tests for _build_repo_setup_script."""

    def test_single_repo_defaults(self):
        repos = {
            "main": RepoConfig(url="https://github.com/user/repo.git"),
        }
        result = _build_repo_setup_script(repos)
        assert "echo '-> Cloning main...'" in result
        assert "git clone --branch main https://github.com/user/repo.git main" in result
        assert "cd main" in result
        assert "pip install --no-cache-dir ." in result
        assert "git checkout" not in result
        assert "cd .." in result

    def test_repo_with_specific_commit(self):
        repos = {
            "lib": RepoConfig(
                url="https://github.com/user/lib.git",
                branch="dev",
                commit="abc123",
            ),
        }
        result = _build_repo_setup_script(repos)
        assert "git checkout abc123" in result
        assert "git clone --branch dev" in result

    def test_repo_without_pip_install(self):
        repos = {
            "data": RepoConfig(
                url="https://github.com/user/data.git",
                pip_install=False,
            ),
        }
        result = _build_repo_setup_script(repos)
        assert "pip install" not in result

    def test_multiple_repos(self):
        repos = {
            "main": RepoConfig(url="https://github.com/user/main.git"),
            "utils": RepoConfig(url="https://github.com/user/utils.git"),
        }
        result = _build_repo_setup_script(repos)
        assert "Cloning main" in result
        assert "Cloning utils" in result
        assert result.count("cd ..") == 2

    def test_empty_repos(self):
        result = _build_repo_setup_script({})
        assert result == ""


class TestBuildEnvExports:
    """Tests for _build_env_exports."""

    def test_single_var(self):
        result = _build_env_exports({"KEY": "value"})
        assert result == 'export KEY="value"'

    def test_multiple_vars(self):
        result = _build_env_exports({"A": "1", "B": "2"})
        assert 'export A="1"' in result
        assert 'export B="2"' in result

    def test_empty_env(self):
        result = _build_env_exports({})
        assert result == ""

    def test_values_with_spaces(self):
        result = _build_env_exports({"PATH": "/opt/bin:/usr/bin"})
        assert 'export PATH="/opt/bin:/usr/bin"' in result


class TestBuildContainerScript:
    """Tests for _build_container_script."""

    def test_generates_heredoc_with_clean_yaml(self):
        clean_yaml = "training:\n  lr: 0.001\n"
        result = _build_container_script(clean_yaml, "", "python train.py")
        assert "cat << 'EOF_CONFIG' > /tmp/clean_config_$$.yaml" in result
        assert clean_yaml in result
        assert "EOF_CONFIG" in result
        assert "rm /tmp/clean_config_$$.yaml" in result

    def test_includes_repo_setup(self):
        clean_yaml = "training:\n  lr: 0.001\n"
        repo_setup = "echo '-> Cloning main...'"
        result = _build_container_script(clean_yaml, repo_setup, "python train.py")
        assert repo_setup in result

    def test_includes_target_command_with_config_arg(self):
        clean_yaml = "key: value\n"
        result = _build_container_script(clean_yaml, "", "python train.py")
        assert "python train.py --config /tmp/clean_config_$$.yaml" in result

    def test_set_e_and_workspace_setup(self):
        result = _build_container_script("", "", "python train.py")
        assert "set -e" in result
        assert 'export PATH="/opt/conda/envs/perception_env/bin:$PATH"' in result
        assert 'WORKSPACE="/tmp/custom_code"' in result
        assert 'mkdir -p "$WORKSPACE"' in result


class TestResolveMounts:
    """Tests for _resolve_mounts."""

    def test_existing_netrc_mounted(self):
        deploy_cfg = DeploymentConfig(
            container=ContainerConfig(
                mount_netrc=True,
                netrc_host_path="/home/user/.netrc",
                netrc_container_path="/root/.netrc",
            ),
        )
        with patch.object(Path, "expanduser", return_value=Path("/home/user/.netrc")):
            with patch.object(Path, "resolve", return_value=Path("/home/user/.netrc")):
                with patch.object(Path, "exists", return_value=True):
                    mounts = _resolve_mounts(deploy_cfg)
        assert "/home/user/.netrc:/root/.netrc" in mounts

    def test_netrc_not_found_warns(self):
        deploy_cfg = DeploymentConfig(
            container=ContainerConfig(
                mount_netrc=True,
                netrc_host_path="~/.netrc",
            ),
        )
        with patch.object(Path, "expanduser", return_value=Path("/home/user/.netrc")):
            with patch.object(Path, "resolve", return_value=Path("/home/user/.netrc")):
                with patch.object(Path, "exists", return_value=False):
                    mounts = _resolve_mounts(deploy_cfg)
        assert len(mounts) == 0

    def test_netrc_disabled(self):
        deploy_cfg = DeploymentConfig(
            container=ContainerConfig(mount_netrc=False),
        )
        mounts = _resolve_mounts(deploy_cfg)
        assert not any(".netrc" in m for m in mounts)

    def test_preserves_existing_mounts(self):
        deploy_cfg = DeploymentConfig(
            container=ContainerConfig(
                mounts=["/data:/data", "/scratch:/scratch"],
                mount_netrc=False,
            ),
        )
        mounts = _resolve_mounts(deploy_cfg)
        assert "/data:/data" in mounts
        assert "/scratch:/scratch" in mounts
        assert len(mounts) == 2


class TestBuildSbatchContent:
    """Tests for _build_sbatch_content."""

    def test_sbatch_directives_generated(self, tmp_path):
        deploy_cfg = _sample_deployment_config()
        content, sbatch_file = _build_sbatch_content(
            deploy_cfg,
            env_exports="",
            container_script="echo hello",
            mounts_str="/data:/data",
            log_dir=tmp_path,
            timestamp="20240101_120000",
        )
        assert "#SBATCH --job-name=train_job" in content
        assert "#SBATCH --partition=gpu" in content
        assert "#SBATCH --time=0-08:00:00" in content
        assert "#SBATCH --nodes=1" in content
        assert "#SBATCH --ntasks=1" in content
        assert "#SBATCH --gpus-per-task=2" in content
        assert "#SBATCH --cpus-per-task=32" in content
        assert "#SBATCH --mem=128G" in content
        assert "#SBATCH --gpu-bind=none" in content

    def test_sbatch_includes_srun_call(self, tmp_path):
        deploy_cfg = _sample_deployment_config()
        content, _ = _build_sbatch_content(
            deploy_cfg,
            env_exports="",
            container_script="echo hello",
            mounts_str="/data:/data",
            log_dir=tmp_path,
            timestamp="20240101_120000",
        )
        assert "srun -K" in content
        assert "--container-image=nvcr.io/nvidia/pytorch:23.10-py3" in content
        assert "--container-mounts=/data:/data" in content

    def test_sbatch_includes_env_exports(self, tmp_path):
        deploy_cfg = _sample_deployment_config()
        content, _ = _build_sbatch_content(
            deploy_cfg,
            env_exports="export WANDB_API_KEY=\"test123\"",
            container_script="echo hello",
            mounts_str="",
            log_dir=tmp_path,
            timestamp="20240101_120000",
        )
        assert "export WANDB_API_KEY=\"test123\"" in content

    def test_sbatch_includes_container_script(self, tmp_path):
        deploy_cfg = _sample_deployment_config()
        content, _ = _build_sbatch_content(
            deploy_cfg,
            env_exports="",
            container_script="echo 'running training'",
            mounts_str="",
            log_dir=tmp_path,
            timestamp="20240101_120000",
        )
        assert "echo 'running training'" in content

    def test_output_path_uses_job_name(self, tmp_path):
        deploy_cfg = DeploymentConfig(
            slurm=SlurmConfig(job_name="my_custom_job"),
        )
        content, _ = _build_sbatch_content(
            deploy_cfg,
            env_exports="",
            container_script="echo hello",
            mounts_str="",
            log_dir=tmp_path,
            timestamp="20240101_120000",
        )
        assert "my_custom_job_%j.out" in content

    def test_sbatch_file_path_generated(self, tmp_path):
        deploy_cfg = _sample_deployment_config()
        _, sbatch_file = _build_sbatch_content(
            deploy_cfg,
            env_exports="",
            container_script="echo hello",
            mounts_str="",
            log_dir=tmp_path,
            timestamp="20240101_120000",
        )
        assert sbatch_file.parent == tmp_path
        assert "train_job_20240101_120000.sbatch" == sbatch_file.name


class TestRunPrintConfigLocally:
    """Tests for _run_print_config_locally."""

    def test_writes_temp_yaml_and_runs_target(self):
        target_args = ["python", "train.py", "--some", "flag"]
        clean_yaml_str = "param: value\n"

        with patch("jsonargparse_slurm.cli.subprocess.run") as mock_run:
            with patch("tempfile.NamedTemporaryFile") as mock_tmp_class:
                mock_file = MagicMock()
                mock_file.name = "/tmp/tmpabc.yaml"
                mock_tmp_class.return_value.__enter__.return_value = mock_file

                with patch("os.path.exists", return_value=True):
                    with patch("os.remove") as mock_remove:
                        result = _run_print_config_locally(target_args, clean_yaml_str)

        assert result == 0
        mock_file.write.assert_called_once_with(clean_yaml_str)
        expected_cmd = target_args + ["--config", "/tmp/tmpabc.yaml", "--print_config"]
        mock_run.assert_called_once_with(expected_cmd, check=True)
        mock_remove.assert_called_once_with("/tmp/tmpabc.yaml")

    def test_handles_subprocess_error(self):
        target_args = ["python", "train.py"]
        clean_yaml_str = "param: value\n"

        with patch("jsonargparse_slurm.cli.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
            with patch("tempfile.NamedTemporaryFile") as mock_tmp_class:
                mock_file = MagicMock()
                mock_file.name = "/tmp/tmpabc.yaml"
                mock_tmp_class.return_value.__enter__.return_value = mock_file

                with patch("os.path.exists", return_value=True):
                    with patch("os.remove"):
                        result = _run_print_config_locally(target_args, clean_yaml_str)

        assert result == 1


class TestDispatchSlurmJob:
    """Tests for _dispatch_slurm_job."""

    def test_writes_sbatch_file_and_submits(self, tmp_path):
        target_args = ["python", "train.py"]
        deploy_cfg = _sample_deployment_config()
        clean_yaml_str = "training:\n  lr: 0.001\n"

        with patch("jsonargparse_slurm.cli.subprocess.run") as mock_run:
            with patch("jsonargparse_slurm.cli.Path.resolve") as mock_log_resolve:
                mock_log_resolve.return_value = tmp_path
                result = _dispatch_slurm_job(target_args, deploy_cfg, clean_yaml_str)

        assert result == 0
        mock_run.assert_called_once()
        assert "sbatch" == mock_run.call_args[0][0][0]

    def test_sbatch_file_contains_clean_yaml(self, tmp_path):
        target_args = ["python", "train.py"]
        deploy_cfg = _sample_deployment_config()
        clean_yaml_str = "training:\n  lr: 0.001\n"

        with patch("jsonargparse_slurm.cli.subprocess.run"):
            with patch("jsonargparse_slurm.cli.Path.resolve") as mock_log_resolve:
                mock_log_resolve.return_value = tmp_path
                with patch("builtins.open", create=True) as mock_open_fn:
                    mock_file = MagicMock()
                    mock_open_fn.return_value.__enter__.return_value = mock_file
                    result = _dispatch_slurm_job(target_args, deploy_cfg, clean_yaml_str)

        assert result == 0
        mock_file.write.assert_called_once()
        written_sbatch = mock_file.write.call_args[0][0]
        assert "clean_config_$$.yaml" in written_sbatch
        assert "training:\n  lr: 0.001" in written_sbatch


class TestMain:
    """Integration tests for the main() function."""

    def _write_config(self, tmp_path, data):
        import yaml
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(data, f)
        return str(config_path)

    def test_print_config_flow(self, tmp_path):
        config_path = self._write_config(tmp_path, _sample_yaml())

        with patch("sys.argv", [
            "jsap-slurm",
            "--config", config_path,
            "--print_config",
            "python", "train.py",
        ]):
            with patch("jsonargparse_slurm.cli.subprocess.run") as mock_run:
                result = main()

        assert result == 0
        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        assert "--print_config" in called_args
        assert "--config" in called_args

    def test_dispatch_flow_generates_sbatch(self, tmp_path):
        config_path = self._write_config(tmp_path, _sample_yaml())

        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            with patch("sys.argv", [
                "jsap-slurm",
                "--config", config_path,
                "python", "train.py",
            ]):
                with patch("jsonargparse_slurm.cli.subprocess.run"):
                    result = main()
        finally:
            os.chdir(original_cwd)

        assert result == 0
        generated = list(tmp_path.glob("logs/train_job_*.sbatch"))
        assert len(generated) == 1
        content = generated[0].read_text()
        assert "#SBATCH --job-name=train_job" in content

    def test_missing_config_file_errors(self):
        with patch("sys.argv", [
            "jsap-slurm",
            "--config", "/nonexistent/config.yaml",
            "python", "train.py",
        ]):
            with patch.object(Path, "exists", return_value=False):
                result = main()

        assert result == 1

    def test_missing_deployment_section_errors(self, tmp_path):
        yaml_data = {"param": "value"}
        config_path = self._write_config(tmp_path, yaml_data)

        with patch("sys.argv", [
            "jsap-slurm",
            "--config", config_path,
            "python", "train.py",
        ]):
            result = main()

        assert result == 1

    def test_cli_overrides_forwarded_to_target(self, tmp_path):
        config_path = self._write_config(tmp_path, _sample_yaml())

        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            with patch("sys.argv", [
                "jsap-slurm",
                "--config", config_path,
                "python", "train.py",
                "--integrator.min_points", "10",
                "--model.arch", "resnet50",
            ]):
                with patch("jsonargparse_slurm.cli.subprocess.run"):
                    result = main()
        finally:
            os.chdir(original_cwd)

        assert result == 0
        generated = list(tmp_path.glob("logs/train_job_*.sbatch"))
        assert len(generated) == 1
        content = generated[0].read_text()
        assert "--integrator.min_points" in content
        assert "resnet50" in content

    def test_deployment_cli_overrides_applied(self, tmp_path):
        config_path = self._write_config(tmp_path, _sample_yaml())

        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            with patch("sys.argv", [
                "jsap-slurm",
                "--deployment.slurm.job_name", "override_job",
                "--config", config_path,
                "python", "train.py",
            ]):
                with patch("jsonargparse_slurm.cli.subprocess.run"):
                    result = main()
        finally:
            os.chdir(original_cwd)

        assert result == 0
        generated = list(tmp_path.glob("logs/override_job_*.sbatch"))
        assert len(generated) == 1
        content = generated[0].read_text()
        assert "#SBATCH --job-name=override_job" in content


class TestYamlCleaning:
    """Tests verifying deployment block removal from YAML."""

    def test_deployment_block_stripped_from_container_script(self, tmp_path):
        target_args = ["python", "train.py"]
        deploy_cfg = _sample_deployment_config()
        clean_yaml_str = "training:\n  lr: 0.001\ndata:\n  path: /data/dataset\n"

        with patch("jsonargparse_slurm.cli.subprocess.run"):
            with patch("jsonargparse_slurm.cli.Path.resolve") as mock_log_resolve:
                mock_log_resolve.return_value = tmp_path
                with patch("builtins.open", create=True) as mock_open_fn:
                    mock_file = MagicMock()
                    mock_open_fn.return_value.__enter__.return_value = mock_file
                    _dispatch_slurm_job(target_args, deploy_cfg, clean_yaml_str)

        written_sbatch = mock_file.write.call_args[0][0]
        heredoc_start = written_sbatch.find("cat << 'EOF_CONFIG'")
        heredoc_end = written_sbatch.find("EOF_CONFIG", heredoc_start)
        heredoc_content = written_sbatch[heredoc_start:heredoc_end]
        assert "deployment" not in heredoc_content
        assert "lr: 0.001" in written_sbatch
        assert "path: /data/dataset" in written_sbatch


class TestNetrcMountingInSbatch:
    """Tests for .netrc mounting in generated SBATCH output."""

    def test_netrc_mount_in_sbatch_when_enabled(self, tmp_path):
        deploy_cfg = DeploymentConfig(
            slurm=SlurmConfig(job_name="netrc_test"),
            container=ContainerConfig(
                image="test:latest",
                mount_netrc=True,
                netrc_host_path="/home/user/.netrc",
                netrc_container_path="/root/.netrc",
            ),
        )

        with patch.object(Path, "expanduser", return_value=Path("/home/user/.netrc")):
            with patch.object(Path, "resolve", return_value=Path("/home/user/.netrc")):
                with patch.object(Path, "exists", return_value=True):
                    content, _ = _build_sbatch_content(
                        deploy_cfg,
                        env_exports="",
                        container_script="echo test",
                        mounts_str="/home/user/.netrc:/root/.netrc",
                        log_dir=tmp_path,
                        timestamp="20240101_120000",
                    )

        assert "/home/user/.netrc:/root/.netrc" in content

    def test_netrc_not_in_sbatch_when_disabled(self, tmp_path):
        deploy_cfg = DeploymentConfig(
            slurm=SlurmConfig(job_name="netrc_test"),
            container=ContainerConfig(mount_netrc=False),
        )

        content, _ = _build_sbatch_content(
            deploy_cfg,
            env_exports="",
            container_script="echo test",
            mounts_str="",
            log_dir=tmp_path,
            timestamp="20240101_120000",
        )

        assert ".netrc" not in content
