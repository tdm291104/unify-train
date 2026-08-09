import pytest
from core.hook.hooks import HookManager, HookContext


def test_register_and_fire():
    hooks = HookManager()
    called = []
    hooks.register("before_train", lambda ctx: called.append("fired"))
    hooks.fire("before_train", HookContext())
    assert called == ["fired"]


def test_multiple_hooks_fire_in_registration_order():
    hooks = HookManager()
    order = []
    hooks.register("after_step", lambda ctx: order.append(1))
    hooks.register("after_step", lambda ctx: order.append(2))
    hooks.fire("after_step", HookContext())
    assert order == [1, 2]


def test_hook_receives_mutable_context():
    hooks = HookManager()
    hooks.register("after_step", lambda ctx: ctx.metrics.update({"flagged": True}))
    ctx = HookContext()
    hooks.fire("after_step", ctx)
    assert ctx.metrics.get("flagged") is True


def test_hook_reads_context_fields():
    hooks = HookManager()
    seen = []
    hooks.register("after_epoch", lambda ctx: seen.append((ctx.epoch, ctx.loss)))
    ctx = HookContext(epoch=2, loss=0.42)
    hooks.fire("after_epoch", ctx)
    assert seen == [(2, 0.42)]


def test_unknown_event_raises():
    hooks = HookManager()
    with pytest.raises(ValueError, match="Unknown hook event"):
        hooks.register("not_an_event", lambda ctx: None)


def test_firing_event_with_no_hooks_does_not_raise():
    HookManager().fire("before_train", HookContext())


def test_all_valid_events_accept_registration():
    hooks = HookManager()
    for event in ["before_train", "after_train", "before_epoch", "after_epoch", "after_step"]:
        hooks.register(event, lambda ctx: None)
