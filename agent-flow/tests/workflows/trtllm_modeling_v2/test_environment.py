# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the resolved launch environment."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import (
    MULTI_TASK_SLURM_SCRUB,
    RunEnvironment,
)


def test_every_command_scrubs_the_allocations_launcher_environment():
    """Otherwise every rank is rank 0 of a world of 1, and the run hangs.

    `trtllm-llmapi-launch` reads `SLURM_PROCID` before
    `OMPI_COMM_WORLD_RANK` and `SLURM_NTASKS` before `OMPI_COMM_WORLD_SIZE`.
    Inside a one-task allocation those are `0` and `1` for every rank, so all
    four become `mgmn_leader_node` and wait forever for workers.

    mpi4py auto-initializes whenever PMI variables are present. A spool script
    runs inside the session's own allocation, so they are, and a lone probe
    would join a world it never asked for.
    """
    env = RunEnvironment(kind="session_spool", python="/p", world_size=4, spool="/s")
    command = env.command("/p x.py")
    for name in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID", "SLURM_NODEID"):
        assert f"-u {name}" in command, name
    assert command.index(MULTI_TASK_SLURM_SCRUB) < command.index("mpirun")
    single = env.command("/p probe.py", single_process=True)
    assert "${!PMIX@}" in single
    assert "exec /p probe.py" in single


def test_a_one_rank_command_finds_the_measured_interpreters_pip_first():
    """The build locates its headers through `pip show tensorrt_llm` when NVRTC compiles at decode.

    A session PATH that leads to another environment's pip fails the first decode step, so a
    one-rank command puts the interpreter's directory first, as a multi-rank launch does.
    """
    env = RunEnvironment(kind="session_spool", python="/w/venv/bin/python", spool="/s")
    command = env.python_command("-m pytest t.py")
    assert 'PATH=/w/venv/bin:"$PATH" /w/venv/bin/python -m pytest t.py' in command
    # Not engine work: still first on PATH, outside the scrub, at any world size.
    for world_size in (1, 4):
        single = RunEnvironment(
            kind="session_spool", python="/w/venv/bin/python", world_size=world_size, spool="/s"
        ).command("/w/venv/bin/python probe.py", single_process=True)
        assert 'PATH=/w/venv/bin:"$PATH" bash -c' in single, world_size
        assert "exec /w/venv/bin/python probe.py" in single, world_size
    # An interpreter found on PATH has no directory to put first.
    assert "PATH=" not in RunEnvironment(kind="local", python="python3").command("x.py")


def test_a_spec_that_states_path_keeps_it_as_stated():
    env = RunEnvironment(
        kind="session_spool", python="/w/venv/bin/python", env={"PATH": "/opt/bin:/usr/bin"}
    )
    for command in (env.command("x.py"), env.command("probe.py", single_process=True)):
        assert "PATH=/opt/bin:/usr/bin" in command
        assert "/w/venv/bin:" not in command
