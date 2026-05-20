"""Tests for the CLI module with mocked SLURM interactions."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml
from jsonargparse import auto_parser

from jsonargparse_slurm.cli import (
    setup_cluster,
    split_by_signature,
    _filter_parser_args,
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
    """YAML with top-level wrapper keys slurm/container (no deployment wrapper)."""
    return {
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
        "training": {
            "lr": 0.001,
            "epochs": 100,
        },
        "data": {
            "path": "/data/dataset",
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


class TestSetupCluster:
    """Tests for setup_cluster."""

    def test_returns_deployment_config(self):
        result = setup_cluster(
            slurm=SlurmConfig(job_name="test"),
            container=ContainerConfig(image="img"),
        )
        assert isinstance(result, DeploymentConfig)
        assert result.slurm.job_name == "test"
        assert result.container.image == "img"

    def test_all_defaults(self):
        result = setup_cluster()
        assert isinstance(result, DeploymentConfig)
        assert result.slurm.job_name == "jsap_job"
        assert result.container.image == "ubuntu:22.04"
        assert result.repos == {}


class TestSplitBySignature:
    """Tests for split_by_signature."""

    def _parser(self):
        return auto_parser(setup_cluster, description="test")

    def test_splits_yaml_keys(self):
        parser = self._parser()
        raw_yaml = _sample_yaml()
        w_args, t_args, w_yaml, t_yaml = split_by_signature(
            parser, [], raw_yaml
        )
        assert "slurm" in w_yaml
        assert "container" in w_yaml
        assert "training" in t_yaml
        assert "data" in t_yaml
        assert "slurm" not in t_yaml
        assert "container" not in t_yaml

    def test_splits_cli_args(self):
        parser = self._parser()
        w_args, t_args, w_yaml, t_yaml = split_by_signature(
            parser,
            [
                "--slurm.job_name", "myjob",
                "--config", "config.yaml",
                "python", "train.py",
                "--integrator.min_points", "10",
            ],
            {},
        )
        assert "--slurm.job_name" in w_args
        assert "--config" in w_args
        assert "config.yaml" in w_args
        assert w_args == [
            "--slurm.job_name", "myjob", "--config", "config.yaml",
        ]
        assert t_args == ["python", "train.py", "--integrator.min_points", "10"]

    def test_handles_deployment_equals_form(self):
        parser = self._parser()
        w_args, t_args, _, _ = split_by_signature(
            parser,
            ["--slurm.job_name=myjob", "python", "train.py"],
            {},
        )
        assert "--slurm.job_name=myjob" in w_args
        assert t_args == ["python", "train.py"]

    def test_print_config_in_wrapper_args(self):
        parser = self._parser()
        w_args, t_args, _, _ = split_by_signature(
            parser,
            ["--print_config", "--config", "config.yaml", "python", "train.py"],
            {},
        )
        assert "--print_config" in w_args
        assert "--config" in w_args
        assert t_args == ["python", "train.py"]

    def test_empty_yaml(self):
        parser = self._parser()
        w_args, t_args, w_yaml, t_yaml = split_by_signature(
            parser, [], {}
        )
        assert w_yaml == {}
        assert t_yaml == {}


class TestFilterParserArgs:
    """Tests for _filter_parser_args."""

    def test_removes_config_flag_and_value(self):
        result = _filter_parser_args([
            "--slurm.job_name", "myjob",
            "--config", "/tmp/config.yaml",
            "--print_config",
        ])
        assert "--config" not in result
        assert "/tmp/config.yaml" not in result
        assert "--print_config" not in result
        assert result == ["--slurm.job_name", "myjob"]

    def test_removes_short_config_flag(self):
        result = _filter_parser_args([
            "-c", "/tmp/config.yaml",
            "--slurm.job_name", "myjob",
        ])
        assert "-c" not in result
        assert "/tmp/config.yaml" not in result
        assert result == ["--slurm.job_name", "myjob"]

    def test_preserves_non_wrapper_args(self):
        result = _filter_parser_args([
            "--slurm.job_name", "myjob",
            "--container.image", "img:v1",
        ])
        assert result == ["--slurm.job_name", "myjob", "--container.image", "img:v1"]


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
        container_cfg = ContainerConfig(
            mount_netrc=True,
            netrc_host_path="/home/user/.netrc",
            netrc_container_path="/root/.netrc",
        )
        with patch.object(Path, "expanduser", return_value=Path("/home/user/.netrc")):
            with patch.object(Path, "resolve", return_value=Path("/home/user/.netrc")):
                with patch.object(Path, "exists", return_value=True):
                    mounts = _resolve_mounts(container_cfg)
        assert "/home/user/.netrc:/root/.netrc" in mounts

    def test_netrc_not_found_warns(self):
        container_cfg = ContainerConfig(
            mount_netrc=True,
            netrc_host_path="~/.netrc",
        )
        with patch.object(Path, "expanduser", return_value=Path("/home/user/.netrc")):
            with patch.object(Path, "resolve", return_value=Path("/home/user/.netrc")):
                with patch.object(Path, "exists", return_value=False):
                    mounts = _resolve_mounts(container_cfg)
        assert len(mounts) == 0

    def test_netrc_disabled(self):
        container_cfg = ContainerConfig(mount_netrc=False)
        mounts = _resolve_mounts(container_cfg)
        assert not any(".netrc" in m for m in mounts)

    def test_preserves_existing_mounts(self):
        container_cfg = ContainerConfig(
            mounts=["/data:/data", "/scratch:/scratch"],
            mount_netrc=False,
        )
        mounts = _resolve_mounts(container_cfg)
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
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(data, f)
        return str(config_path)

    def test_print_config_flow(self, tmp_path, capsys):
        config_path = self._write_config(tmp_path, _sample_yaml())

        with patch("sys.argv", [
            "jsap-slurm",
            "--config", config_path,
            "python", "train.py",
            "--print_config",
        ]):
            with patch("jsonargparse_slurm.cli.subprocess.run") as mock_run:
                result = main()

        assert result == 0
        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        assert "--print_config" in called_args
        assert "--config" in called_args
        stdout = capsys.readouterr().out
        assert "job_name: train_job" in stdout
        assert "partition: gpu" in stdout

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

    def test_no_wrapper_keys_uses_defaults(self, tmp_path):
        yaml_data = {"training": {"lr": 0.001}}
        config_path = self._write_config(tmp_path, yaml_data)

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
        generated = list(tmp_path.glob("logs/jsap_job_*.sbatch"))
        assert len(generated) == 1

    def test_missing_config_file_errors(self, tmp_path):
        config_path = str(tmp_path / "nonexistent.yaml")

        with patch("sys.argv", [
            "jsap-slurm",
            "--config", config_path,
            "python", "train.py",
        ]):
            with patch.object(Path, "exists", return_value=False):
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
                "--slurm.job_name", "override_job",
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

    def test_deployment_equals_form_cli_override(self, tmp_path):
        config_path = self._write_config(tmp_path, _sample_yaml())

        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            with patch("sys.argv", [
                "jsap-slurm",
                "--slurm.job_name=override_job",
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


    def test_nothing_passed_uses_all_defaults(self, tmp_path):
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            with patch("sys.argv", [
                "jsap-slurm",
                "python", "train.py",
            ]):
                with patch("jsonargparse_slurm.cli.subprocess.run"):
                    result = main()
        finally:
            os.chdir(original_cwd)

        assert result == 0
        generated = list(tmp_path.glob("logs/jsap_job_*.sbatch"))
        assert len(generated) == 1
        content2 = generated[0].read_text()
        assert "#SBATCH --job-name=jsap_job" in content2

    def test_only_cli_overrides_no_yaml(self, tmp_path):
        original_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            with patch("sys.argv", [
                "jsap-slurm",
                "--slurm.job_name", "cli_only_job",
                "--slurm.partition", "debug",
                "python", "train.py",
            ]):
                with patch("jsonargparse_slurm.cli.subprocess.run"):
                    result = main()
        finally:
            os.chdir(original_cwd)

        assert result == 0
        generated = list(tmp_path.glob("logs/cli_only_job_*.sbatch"))
        assert len(generated) == 1
        content3 = generated[0].read_text()
        assert "#SBATCH --job-name=cli_only_job" in content3
        assert "#SBATCH --partition=debug" in content3


class TestYamlCleaning:
    """Tests verifying wrapper keys are stripped from the target YAML."""

    def test_wrapper_keys_stripped_from_container_script(self, tmp_path):
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
        assert "slurm" not in heredoc_content
        assert "container" not in heredoc_content
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
