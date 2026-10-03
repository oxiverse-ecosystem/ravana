"""Contract tests for the tool safety-class declaration (backlog task 10).

Two drift modes must be impossible:

1. A tool registered WITHOUT a declared safety class.
2. A tool whose declared safety class disagrees with the guards actually
   applied to it (a guard declared but not enforced, or a guard enforced that
   the class does not declare).

Both are checked mechanically here, so the declaration cannot rot silently.
"""
from __future__ import annotations

import pytest

from ravana.agent.tool_registry import (
    SAFETY_CLASSES,
    Tool,
    ToolCall,
    ToolRegistry,
    _GUARD_IMPLS,
    _GUARD_PROBES,
    build_registry,
    register_safety_class,
    safety_class,
)
from ravana.agent.tool_registry import SafetyClass


def test_every_registered_tool_declares_a_safety_class():
    for name, tool in build_registry().items():
        assert tool.safety_class, f"tool '{name}' has no declared safety class"
        assert tool.safety_class in SAFETY_CLASSES, (
            f"tool '{name}' declares unknown class {tool.safety_class!r}")


def test_tool_without_safety_class_cannot_be_constructed():
    with pytest.raises(ValueError):
        Tool(name="rogue", description="undeclared", run=lambda a: "x")
    with pytest.raises(ValueError):
        Tool(name="rogue", description="undeclared",
             safety_class="no_such_class", run=lambda a: "x")


def test_class_projects_onto_destructive_flag_and_guards():
    for tool in build_registry().values():
        cls = safety_class(tool.safety_class)
        assert tool.is_destructive is cls.destructive
        assert tool.required_guards == tuple(cls.guards)


def test_every_declared_guard_exists_and_has_a_probe():
    for cls in SAFETY_CLASSES.values():
        for guard in cls.guards:
            assert guard in _GUARD_IMPLS, f"guard {guard!r} has no implementation"
            assert guard in _GUARD_PROBES, f"guard {guard!r} has no proof probe"


def test_every_guard_implementation_actually_blocks_its_probe():
    """The anti-drift check: a declared guard must really reject."""
    for guard, probe in _GUARD_PROBES.items():
        with pytest.raises((PermissionError, ValueError)):
            _GUARD_IMPLS[guard](probe)


def test_declared_guards_are_enforced_on_the_tool_path():
    """Each tool must actually block the probe of every guard it declares."""
    reg = ToolRegistry()
    for name, tool in reg.tools.items():
        for guard in tool.required_guards:
            probe = _GUARD_PROBES[guard]
            result = reg.execute(ToolCall(tool=name, arg=probe, reason="test"))
            assert "BLOCKED" in result or "error" in result, (
                f"tool '{name}' declares guard '{guard}' but "
                f"probe was not blocked: {result!r}")


def test_registry_wide_pattern_guard_applies_to_every_tool():
    reg = ToolRegistry()
    for name in reg.names():
        result = reg.execute(ToolCall(tool=name, arg="rm -rf /", reason="test"))
        assert "BLOCKED" in result, f"registry-wide guard missed tool '{name}'"


def test_destructive_flag_is_derived_not_authored():
    """is_destructive is a projection, so a caller cannot pass its own value."""
    tool = Tool(name="probe", description="probe", safety_class="repo_write",
                is_destructive=False, run=lambda a: "x")
    assert tool.is_destructive is True


def test_runtime_registered_safety_class_is_usable():
    register_safety_class(
        SafetyClass(name="unit_test_class", guards=("public_url",),
                    destructive=False), replace=True)
    try:
        tool = Tool(name="unit_tool", description="probe",
                    safety_class="unit_test_class", run=lambda a: "ok")
        assert tool.required_guards == ("public_url",)
        assert tool.is_destructive is False
    finally:
        SAFETY_CLASSES.pop("unit_test_class", None)


def test_registering_a_class_with_an_unknown_guard_fails():
    with pytest.raises(ValueError):
        register_safety_class(
            SafetyClass(name="bad_class", guards=("no_such_guard",)),
            replace=True)
    assert "bad_class" not in SAFETY_CLASSES


def test_registering_a_duplicate_class_fails_without_replace():
    with pytest.raises(ValueError):
        register_safety_class(SafetyClass(name="network_read"))


def test_safety_class_lookup_fails_closed():
    with pytest.raises(ValueError):
        safety_class("definitely_not_declared")
