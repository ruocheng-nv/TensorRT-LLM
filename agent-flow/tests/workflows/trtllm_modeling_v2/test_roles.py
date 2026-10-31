# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-role backend routing for the workflow's roles."""

from __future__ import annotations

import pytest

from agent_flow.logger import get_logger
from agent_flow.workflows.trtllm_modeling_v2.agents import roles
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import (
    PROMPTS_DIRNAME,
    build_trtllm_modeling_v2_prompts,
)
from agent_flow.workflows.trtllm_modeling_v2.operators.inventory import Inventory
from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap

from .test_workflow import _workflow


@pytest.fixture(autouse=True)
def _isolated_driving_agent(monkeypatch):
    for name in (roles.CODEX_SESSION_ENV, roles.CLAUDE_SESSION_ENV):
        monkeypatch.delenv(name, raising=False)


def test_a_claude_role_keeps_its_stop_hook_and_a_codex_role_gets_none(tmp_path):
    """The hook refuses a turn that wrote nothing; the file is validated when read.

    Codex refuses per-client Claude SDK hooks, so a Stop hook would fail the turn.
    """
    workflow = _workflow(tmp_path)
    kb = {"kb": {"type": "http", "url": "https://kb.test/mcp"}}
    workflow.task_data = {
        "agents": {
            "roles": {
                "assembler": {"reasoning_effort": "high"},
                "planner": {"backend": "codex", "extra_mcp_servers": kb},
            }
        }
    }
    layer = workflow._make_agent("assembler", "system")
    backend = layer.config.backend
    assert (backend.kind, backend.model, backend.reasoning_effort) == (
        "claude-code",
        "claude-test-model",
        "high",
    )
    assert backend.hooks and backend.hooks["Stop"]
    assert backend.cwd == workflow.workspace
    assert layer.config.required_tools == ()
    assert backend.codex_home is None and backend.codex_sqlite_home is None

    layer = workflow._make_agent("planner", "system")
    backend = layer.config.backend
    assert (backend.kind, backend.model) == ("codex", "codex-test-model")
    assert backend.hooks is None
    assert backend.extra_mcp_servers == kb
    assert layer.config.required_tools == ()
    # The run's own Codex home, and its databases off the shared file systems.
    assert backend.codex_home == (workflow.workspace / "codex-home").absolute()
    assert backend.codex_sqlite_home.is_dir() and backend.codex_sqlite_home != backend.codex_home
    # Without the user's plugins, and, once the agent probe has read them, their skills.
    assert backend.plugins is False and backend.disabled_skills == ()
    workflow._foreign_skills = {"codex": ("imagegen", "cluster-jobs")}
    backend = workflow._make_agent("planner", "system").config.backend
    assert backend.disabled_skills == ("imagegen", "cluster-jobs")


def _skill(root, name):
    (root / name).mkdir(parents=True)
    (root / name / "SKILL.md").write_text(f"---\nname: {name}\n---\n")


def test_the_operators_extra_skills_are_materialized_and_count_as_the_runs_own(
    tmp_path, monkeypatch
):
    """A tool the roles reach through extra_mcp_servers can bring its skills; the probe must not turn them off."""
    extra = tmp_path / "extra"
    _skill(extra, "tool-authoring")
    _skill(extra, "tool-schemas")
    (extra / "notes").mkdir()  # no SKILL.md: not a skill
    monkeypatch.setenv(roles.EXTRA_SKILLS_ENV, str(extra))
    assert [p.name for p in roles.extra_skill_dirs()] == ["tool-authoring", "tool-schemas"]
    assert {"tool-authoring", "tool-schemas"} <= set(roles.packaged_skills())

    workflow = _workflow(tmp_path)
    skills = workflow._materialize_skills()
    assert (skills / "tool-schemas" / "SKILL.md").is_file()
    assert (skills / "trtllm-modeling-v2-entry").is_dir()
    assert (workflow.workspace / ".agents" / "skills" / "tool-authoring").is_dir()


def test_an_extra_skill_named_like_another_is_refused(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    _skill(first, "tool")
    _skill(second, "tool")
    monkeypatch.setenv(roles.EXTRA_SKILLS_ENV, f"{first}:{second}")
    with pytest.raises(ValueError, match="given twice"):
        roles.extra_skill_dirs()
    _skill(tmp_path / "c", "trtllm-modeling-v2-entry")
    monkeypatch.setenv(roles.EXTRA_SKILLS_ENV, str(tmp_path / "c"))
    with pytest.raises(ValueError, match="given twice"):
        roles.extra_skill_dirs()


def test_an_extra_skills_path_that_gives_no_skill_is_refused(tmp_path, monkeypatch):
    """A skill named directly, or a directory holding none, would give the roles nothing silently."""
    _skill(tmp_path / "extra", "tool")
    monkeypatch.setenv(roles.EXTRA_SKILLS_ENV, str(tmp_path / "extra" / "tool"))
    with pytest.raises(ValueError, match="pass its parent directory"):
        roles.extra_skill_dirs()
    (tmp_path / "empty" / "notes").mkdir(parents=True)
    monkeypatch.setenv(roles.EXTRA_SKILLS_ENV, str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="holds no skill"):
        roles.extra_skill_dirs()


def test_a_brief_that_is_missing_or_empty_is_refused(tmp_path, monkeypatch):
    assert roles.role_brief() == ""
    monkeypatch.setenv(roles.ROLE_BRIEF_ENV, str(tmp_path / "missing.txt"))
    with pytest.raises(ValueError, match="cannot read"):
        roles.role_brief()
    (tmp_path / "blank.txt").write_text("\n  \n")
    monkeypatch.setenv(roles.ROLE_BRIEF_ENV, str(tmp_path / "blank.txt"))
    with pytest.raises(ValueError, match="is empty"):
        roles.role_brief()


def test_the_operators_brief_reaches_every_role_through_the_prompt_bundle(tmp_path, monkeypatch):
    """The bundle is what is dumped and what each turn records: the brief must be in it."""
    bundle = build_trtllm_modeling_v2_prompts(notes="Mind the cache.", brief="Use the tool.")
    for role in roles.ROLES:
        prompt = getattr(bundle, role)
        assert prompt.endswith("\n\nUse the tool."), role
        assert prompt.count("Use the tool.") == 1, role
    # A layer's system prompt is its bundle entry as given: the brief is not added twice.
    brief = tmp_path / "brief.txt"
    brief.write_text("Use the tool.\n")
    monkeypatch.setenv(roles.ROLE_BRIEF_ENV, str(brief))
    workflow = _workflow(tmp_path)
    assert workflow._make_agent("planner", "system").config.system_prompt == "system"


def test_preflight_dumps_the_brief_in_every_roles_prompt(tmp_path, task_file, monkeypatch):
    brief = tmp_path / "brief.txt"
    brief.write_text("Use the tool when it helps.\n")
    monkeypatch.setenv(roles.ROLE_BRIEF_ENV, str(brief))
    monkeypatch.setattr(bootstrap, "run_inventory_probe", lambda *a, **k: Inventory())
    workflow = _workflow(tmp_path)
    log = get_logger().console
    assert bootstrap._run_preflight(workflow, workflow._init_state(task_file, log), log)
    for role in roles.ROLES:
        prompt = getattr(workflow.prompts, role)
        assert prompt.count("\n\nUse the tool when it helps.") == 1, role
        assert (workflow.workspace / PROMPTS_DIRNAME / f"{role}.md").read_text() == prompt, role
