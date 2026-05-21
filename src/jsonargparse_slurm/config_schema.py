"""Configuration dataclasses for the deployment block in the YAML config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SlurmConfig:
    """SLURM resource allocation parameters for the job."""

    job_name: str = "jsap_job"
    partition: str = "batch"
    time: str = "0-04:00:00"
    nodes: int = 1
    ntasks: int = 1
    gpus_per_task: int = 1
    cpus_per_task: int = 16
    mem: str = "64G"
    gpu_bind: str = "none"
    mail_user: Optional[str] = None
    mail_type: Optional[str] = None
    ssh_remote: Optional[str] = None


@dataclass
class ContainerConfig:
    """Container image and runtime configuration for Enroot/Pyxis."""

    image: str = "ubuntu:22.04"
    mounts: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    mount_netrc: bool = True
    netrc_host_path: str = "~/.netrc"
    netrc_container_path: str = "/root/.netrc"


@dataclass
class RepoConfig:
    """Git repository cloning and installation configuration."""

    url: str = ""
    branch: str = "main"
    commit: str = "HEAD"
    pip_install: bool = True


@dataclass
class DeploymentConfig:
    """Top-level deployment configuration encapsulating SLURM, container, and repo settings."""

    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    container: ContainerConfig = field(default_factory=ContainerConfig)
    repos: Dict[str, RepoConfig] = field(default_factory=dict)
