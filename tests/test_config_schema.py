"""Tests for the configuration schema dataclasses."""

from jsonargparse_slurm.config_schema import (
    ContainerConfig,
    DeploymentConfig,
    RepoConfig,
    SlurmConfig,
)


class TestSlurmConfig:
    """Tests for SlurmConfig."""

    def test_defaults(self):
        cfg = SlurmConfig()
        assert cfg.job_name == "jsap_job"
        assert cfg.partition == "batch"
        assert cfg.time == "0-04:00:00"
        assert cfg.nodes == 1
        assert cfg.ntasks == 1
        assert cfg.gpus_per_task == 1
        assert cfg.cpus_per_task == 16
        assert cfg.mem == "64G"
        assert cfg.gpu_bind == "none"

    def test_custom_values(self):
        cfg = SlurmConfig(
            job_name="test_job",
            partition="gpu",
            time="1-00:00:00",
            nodes=2,
            ntasks=4,
            gpus_per_task=2,
            cpus_per_task=32,
            mem="128G",
            gpu_bind="closest",
        )
        assert cfg.job_name == "test_job"
        assert cfg.partition == "gpu"
        assert cfg.time == "1-00:00:00"
        assert cfg.nodes == 2
        assert cfg.ntasks == 4
        assert cfg.gpus_per_task == 2
        assert cfg.cpus_per_task == 32
        assert cfg.mem == "128G"
        assert cfg.gpu_bind == "closest"


class TestContainerConfig:
    """Tests for ContainerConfig."""

    def test_defaults(self):
        cfg = ContainerConfig()
        assert cfg.image == "ubuntu:22.04"
        assert cfg.mounts == []
        assert cfg.env == {}
        assert cfg.mount_netrc is True
        assert cfg.netrc_host_path == "~/.netrc"
        assert cfg.netrc_container_path == "/root/.netrc"

    def test_custom_values(self):
        cfg = ContainerConfig(
            image="nvcr.io/nvidia/pytorch:23.10-py3",
            mounts=["/data:/data", "/scratch:/scratch"],
            env={"CUDA_VISIBLE_DEVICES": "0,1"},
            mount_netrc=False,
            netrc_host_path="/custom/.netrc",
            netrc_container_path="/home/user/.netrc",
        )
        assert cfg.image == "nvcr.io/nvidia/pytorch:23.10-py3"
        assert cfg.mounts == ["/data:/data", "/scratch:/scratch"]
        assert cfg.env == {"CUDA_VISIBLE_DEVICES": "0,1"}
        assert cfg.mount_netrc is False
        assert cfg.netrc_host_path == "/custom/.netrc"
        assert cfg.netrc_container_path == "/home/user/.netrc"


class TestRepoConfig:
    """Tests for RepoConfig."""

    def test_defaults(self):
        cfg = RepoConfig()
        assert cfg.url == ""
        assert cfg.branch == "main"
        assert cfg.commit == "HEAD"
        assert cfg.pip_install is True

    def test_custom_values(self):
        cfg = RepoConfig(
            url="https://github.com/user/repo.git",
            branch="dev",
            commit="abc123",
            pip_install=False,
        )
        assert cfg.url == "https://github.com/user/repo.git"
        assert cfg.branch == "dev"
        assert cfg.commit == "abc123"
        assert cfg.pip_install is False


class TestDeploymentConfig:
    """Tests for DeploymentConfig."""

    def test_defaults(self):
        cfg = DeploymentConfig()
        assert isinstance(cfg.slurm, SlurmConfig)
        assert isinstance(cfg.container, ContainerConfig)
        assert cfg.repos == {}

    def test_nested_config_values(self):
        cfg = DeploymentConfig(
            slurm=SlurmConfig(job_name="nested_job", partition="gpu"),
            container=ContainerConfig(image="custom:latest"),
            repos={
                "main": RepoConfig(url="https://github.com/user/repo.git"),
            },
        )
        assert cfg.slurm.job_name == "nested_job"
        assert cfg.slurm.partition == "gpu"
        assert cfg.slurm.time == "0-04:00:00"
        assert cfg.container.image == "custom:latest"
        assert "main" in cfg.repos
        assert cfg.repos["main"].url == "https://github.com/user/repo.git"
