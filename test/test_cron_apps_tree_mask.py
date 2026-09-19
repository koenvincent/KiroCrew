"""The installed-apps tree mask a script cron spawns under.

``.app_secret`` is a bearer credential: ``dashboard.token_auth.validate_app_secret``
exchanges it for that app's scoped token. A cron script child is masked from every
app's copy at the spawn that needs it, because the sandbox's own crew-home leaf set
does not cover them and the ``cc`` profile leaves them readable.

These pin the property that a SNAPSHOT of app directories cannot have: an app
installed while a long-running script cron is still executing is covered too,
because the mask names the containing directory rather than names read out of it.
"""

import os

import pytest

from kiro_crew import cron_script as cs
from kiro_crew import sandbox as sb


@pytest.fixture()
def crew_home(tmp_path, monkeypatch):
    """A crew home holding two installed apps, each with a secret."""
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
    apps = tmp_path / "apps"
    (apps / "appA").mkdir(parents=True)
    (apps / "appA" / ".app_secret").write_text("SECRET-A", encoding="utf-8")
    (apps / "appA" / "job.py").write_text("def run(ctx):\n    return None\n", encoding="utf-8")
    (apps / "appB" / "crons").mkdir(parents=True)
    (apps / "appB" / ".app_secret").write_text("SECRET-B", encoding="utf-8")
    (apps / "appB" / "crons" / "job.py").write_text(
        "def run(ctx):\n    return None\n", encoding="utf-8"
    )
    return tmp_path


class TestAppsTreeMask:
    def test_the_mask_names_the_tree_not_one_leaf_per_app(self, crew_home):
        """The whole ``apps/`` directory, so the set does not depend on enumeration."""
        apps = (crew_home / "apps").resolve()
        assert cs._app_secret_masks() == (str(apps),)

    def test_an_app_installed_after_the_mask_was_computed_is_covered(self, crew_home):
        """The property a per-app leaf snapshot cannot have.

        A mask is applied to the paths named at spawn and is never recomputed for a
        live child, so a leaf list built by walking ``apps/`` leaves an app installed
        during a long run readable for the rest of that child's life.
        """
        masks = cs._app_secret_masks()
        later = crew_home / "apps" / "installed-mid-run"
        later.mkdir(parents=True)
        later.joinpath(".app_secret").write_text("SECRET-C", encoding="utf-8")

        assert str(later.joinpath(".app_secret").resolve()).startswith(masks[0] + os.sep)

    def test_an_absent_apps_tree_is_created_and_masked(self, tmp_path, monkeypatch):
        """Replaces a pin that asserted an absent tree yields NO mask.

        That pin encoded the reasoning that there is no directory to mask and no
        secret to read, which holds at spawn time and stops holding the moment the
        first install lands mid-run -- the exact case the tree scope exists for. The
        Linux launcher's mask loop is guarded on ``isdir``, so a target that does not
        exist yet gets no bind over it and the directory created later is plainly
        visible to the running child.
        """
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        apps = tmp_path / "apps"
        assert not apps.exists()

        masks = cs._app_secret_masks()

        assert masks == (str(apps.resolve()),)
        assert apps.is_dir()

    def test_a_first_ever_install_during_the_run_is_covered(self, tmp_path, monkeypatch):
        """The finding this reversal closes, stated as behaviour rather than as code.

        A home where no app has ever been installed, an active script cron, and the
        first-ever install landing during that run: without the tree the mask was
        empty, so that app's ``.app_secret`` stayed readable for the rest of the
        child's life.
        """
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        masks = cs._app_secret_masks()
        assert masks is not None

        first = tmp_path / "apps" / "first-app"
        first.mkdir(parents=True)
        first.joinpath(".app_secret").write_text("SECRET-FIRST", encoding="utf-8")

        assert str(first.joinpath(".app_secret").resolve()).startswith(masks[0] + os.sep)

    def test_a_tree_that_cannot_be_created_refuses_rather_than_masking_nothing(
        self, tmp_path, monkeypatch
    ):
        """``None``, not ``()``: the control was requested and could not be built.

        A plain file squatting the name is the reachable shape -- ``mkdir`` reports it
        as ``FileExistsError`` even under ``exist_ok``, because the existing path is
        not a directory. Returning an empty mask here would launch a child that can
        read every app's bearer credential with nothing recording the skip.
        """
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        (tmp_path / "apps").write_text("not a directory", encoding="utf-8")

        assert cs._app_secret_masks() is None


class TestBundleScriptWindow:
    def test_an_operator_cron_gets_no_window(self, crew_home):
        """Its script is not in the tree, so the tree stays wholly masked."""
        masks = cs._app_secret_masks()
        assert cs._bundle_script_window(str(crew_home / "crons" / "job.py"), masks) == ()

    def test_a_bundle_cron_keeps_its_own_script_directory(self, crew_home):
        """The launcher imports the script from that directory, so it must be reachable."""
        masks = cs._app_secret_masks()
        script = crew_home / "apps" / "appB" / "crons" / "job.py"

        window = cs._bundle_script_window(str(script), masks)

        assert window == (str(script.parent.resolve()),)

    def test_the_window_leaves_a_foreign_app_secret_outside_it(self, crew_home):
        """The escalation that matters: one app's cron reading ANOTHER app's token."""
        masks = cs._app_secret_masks()
        script = crew_home / "apps" / "appB" / "crons" / "job.py"
        window = cs._bundle_script_window(str(script), masks)[0]

        foreign = str((crew_home / "apps" / "appA" / ".app_secret").resolve())
        assert not foreign.startswith(window + os.sep)

    def test_a_script_below_its_bundle_root_keeps_its_own_secret_outside_too(self, crew_home):
        """Free consequence of windowing the script's directory rather than the root."""
        masks = cs._app_secret_masks()
        script = crew_home / "apps" / "appB" / "crons" / "job.py"
        window = cs._bundle_script_window(str(script), masks)[0]

        own = str((crew_home / "apps" / "appB" / ".app_secret").resolve())
        assert not own.startswith(window + os.sep)

    def test_the_sandbox_admits_the_window_as_a_proper_descendant(self, crew_home):
        """``_private_window_spellings`` is what decides, so assert against it.

        It admits only a proper descendant of a hidden directory and refuses one that
        EQUALS a hidden target, which is what keeps this from being a mask lift.
        """
        masks = cs._app_secret_masks()
        script = crew_home / "apps" / "appB" / "crons" / "job.py"
        window = cs._bundle_script_window(str(script), masks)

        assert sb._private_window_spellings(window, [masks[0]]) == [window[0]]

    def test_the_tree_itself_is_refused_as_a_window(self, crew_home):
        """A window equal to the hidden target would be a mask lift by another name."""
        masks = cs._app_secret_masks()

        assert sb._private_window_spellings((masks[0],), [masks[0]]) == []

    def test_no_window_without_a_mask(self, crew_home):
        """With nothing hidden there is nothing to re-expose."""
        script = crew_home / "apps" / "appB" / "crons" / "job.py"
        assert cs._bundle_script_window(str(script), ()) == ()


class TestBundleDataWindow:
    """``app_data_dir`` resolves INSIDE the masked tree, so it needs its own window.

    The script window covers it only by accident of layout: a script at the bundle
    root gets a window containing ``data/``, while the nested ``crons/job.py``
    spelling gets a window beside it. Without this, that layout's durable writes
    land in the masked mount and vanish at exit.
    """

    def test_a_nested_script_gets_its_apps_data_directory(self, crew_home):
        """The reachable loss: a window beside ``data/`` rather than around it."""
        masks = cs._app_secret_masks()
        script = crew_home / "apps" / "appB" / "crons" / "job.py"
        script_window = cs._bundle_script_window(str(script), masks)

        window = cs._bundle_data_window(str(script), masks, script_window)

        assert window == (str((crew_home / "apps" / "appB" / "data").resolve()),)

    def test_the_data_window_agrees_with_what_apps_actually_write_to(self, crew_home):
        """Delegated to ``app_data_dir`` so the window cannot drift from the writes."""
        from kiro_crew.apps.manager import app_data_dir

        masks = cs._app_secret_masks()
        script = crew_home / "apps" / "appB" / "crons" / "job.py"
        script_window = cs._bundle_script_window(str(script), masks)

        window = cs._bundle_data_window(str(script), masks, script_window)

        assert window == (str(app_data_dir("appB").resolve()),)

    def test_a_root_layout_script_gets_no_second_window(self, crew_home):
        """Its script window already contains ``data/``, so a nested one is redundant."""
        masks = cs._app_secret_masks()
        script = crew_home / "apps" / "appA" / "job.py"
        script_window = cs._bundle_script_window(str(script), masks)

        assert cs._bundle_data_window(str(script), masks, script_window) == ()

    def test_neither_window_exposes_the_apps_own_secret_in_the_nested_layout(self, crew_home):
        """Covering the data dir must not cost the property the mask buys."""
        masks = cs._app_secret_masks()
        script = crew_home / "apps" / "appB" / "crons" / "job.py"
        script_window = cs._bundle_script_window(str(script), masks)
        windows = script_window + cs._bundle_data_window(str(script), masks, script_window)

        own = str((crew_home / "apps" / "appB" / ".app_secret").resolve())
        for window in windows:
            assert not own.startswith(window.rstrip(os.sep) + os.sep)

    def test_no_window_reaches_a_foreign_apps_data(self, crew_home):
        """One app's cron writing into ANOTHER app's state is the escalation here.

        A read-only leak would be bad enough; a private window is read-WRITE, which
        is why the owner is read from the script's own path.
        """
        masks = cs._app_secret_masks()
        script = crew_home / "apps" / "appB" / "crons" / "job.py"
        script_window = cs._bundle_script_window(str(script), masks)
        windows = script_window + cs._bundle_data_window(str(script), masks, script_window)

        foreign = str((crew_home / "apps" / "appA" / "data").resolve())
        for window in windows:
            assert not foreign.startswith(window.rstrip(os.sep) + os.sep)
            assert foreign != window

    def test_an_operator_cron_gets_no_data_window(self, crew_home):
        """Its script is outside the tree, so no owner can be read and none is needed."""
        masks = cs._app_secret_masks()
        script = crew_home / "crons" / "job.py"

        assert cs._bundle_data_window(str(script), masks, ()) == ()

    def test_a_path_with_no_app_component_creates_nothing(self, crew_home):
        """It belongs to no app, and ``app_data_dir`` CREATES whatever it is handed.

        Driven with a path that does NOT exist, which is the only shape where the
        component count is what decides. An existing FILE directly in the tree root
        is already handled by the ``OSError`` arm, because ``app_data_dir`` would try
        to mkdir beneath a file -- so a test using one passes with the count check
        removed and proves nothing. The absent path is reachable as a race:
        ``resolve_script_path`` confirms the file exists, and it can be deleted
        between that check and this call. Without the count check the first component
        is the filename, so a spawn would silently mkdir ``apps/loose.py/data``.
        """
        masks = cs._app_secret_masks()
        loose = crew_home / "apps" / "loose.py"
        assert not loose.exists()

        assert cs._bundle_data_window(str(loose), masks, ()) == ()
        assert not (crew_home / "apps" / "loose.py").exists()

    def test_an_existing_file_in_the_tree_root_also_yields_no_window(self, crew_home):
        """The same outcome by the other arm, so both paths are pinned."""
        masks = cs._app_secret_masks()
        loose = crew_home / "apps" / "loose.py"
        loose.write_text("def run(ctx):\n    return None\n", encoding="utf-8")

        assert cs._bundle_data_window(str(loose), masks, ()) == ()
        assert loose.is_file()

    def test_no_data_window_without_a_mask(self, crew_home):
        """With nothing hidden there is nothing to re-expose."""
        script = crew_home / "apps" / "appB" / "crons" / "job.py"
        assert cs._bundle_data_window(str(script), (), ()) == ()

    def test_the_sandbox_admits_both_windows_as_proper_descendants(self, crew_home):
        """``_private_window_spellings`` decides, so assert the pair against it."""
        masks = cs._app_secret_masks()
        script = crew_home / "apps" / "appB" / "crons" / "job.py"
        script_window = cs._bundle_script_window(str(script), masks)
        windows = script_window + cs._bundle_data_window(str(script), masks, script_window)

        assert sb._private_window_spellings(windows, [masks[0]]) == list(windows)
