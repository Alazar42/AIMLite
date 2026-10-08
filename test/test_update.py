"""Unit tests for the aimlite update CLI command."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from aimlite.cli.commands.update import (
    detect_environment_description,
    get_installed_version,
    get_latest_pypi_version,
    is_virtual_env,
    parse_version,
    resolve_candidate_commands,
    run_update,
)


class TestVersionParsing:
    def test_parse_version_ordering(self):
        assert parse_version("2.0.0") < parse_version("2.1.0")
        assert parse_version("2.1.0") < parse_version("2.1.1")
        assert parse_version("v2.1.1") == parse_version("2.1.1")
        assert parse_version("2.1.0") == parse_version("2.1.0")

    def test_parse_version_empty(self):
        assert parse_version(None) == ()
        assert parse_version("") == ()


class TestCandidateResolution:
    def test_candidate_commands_in_venv(self):
        with patch("aimlite.cli.commands.update.is_virtual_env", return_value=True):
            with patch("shutil.which", return_value=None):
                cmds = resolve_candidate_commands("aimlite")
                assert len(cmds) >= 1
                assert "--user" not in cmds[0]
                assert cmds[0] == [sys.executable, "-m", "pip", "install", "--upgrade", "aimlite"]

    def test_candidate_commands_outside_venv(self):
        with patch("aimlite.cli.commands.update.is_virtual_env", return_value=False):
            cmds = resolve_candidate_commands("aimlite")
            # First attempt must try --user
            assert "--user" in cmds[0]
            # Second attempt tries --user --break-system-packages
            assert "--user" in cmds[1] and "--break-system-packages" in cmds[1]
            # Third attempt tries global
            assert "--user" not in cmds[2]

    def test_candidate_commands_explicit_user(self):
        cmds = resolve_candidate_commands("aimlite", user_flag=True)
        assert all("--user" in cmd for cmd in cmds)

    def test_candidate_commands_explicit_global(self):
        cmds = resolve_candidate_commands("aimlite", user_flag=False)
        assert all("--user" not in cmd for cmd in cmds)

    def test_candidate_commands_target_version(self):
        cmds = resolve_candidate_commands("aimlite==2.1.1", user_flag=True)
        assert any("aimlite==2.1.1" in cmd for cmd in cmds)


class TestRunUpdateFlow:
    @patch("aimlite.cli.commands.update.get_installed_version", return_value="2.1.1")
    @patch("aimlite.cli.commands.update.get_latest_pypi_version", return_value="2.1.1")
    @patch("subprocess.run")
    def test_already_up_to_date(self, mock_run, mock_latest, mock_installed, capsys):
        code = run_update(force=False)
        assert code == 0
        mock_run.assert_not_called()
        captured = capsys.readouterr().out
        assert "already up to date" in captured.lower()
        assert "v2.1.1" in captured

    @patch("aimlite.cli.commands.update.get_installed_version")
    @patch("aimlite.cli.commands.update.get_latest_pypi_version", return_value="2.1.1")
    @patch("subprocess.run")
    def test_upgrades_when_outdated(self, mock_run, mock_latest, mock_installed, capsys):
        # First call installed is 2.0.0, after install fresh call is 2.1.1
        mock_installed.side_effect = ["2.0.0", "2.1.1"]
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        code = run_update()
        assert code == 0
        mock_run.assert_called()
        captured = capsys.readouterr().out
        assert "Upgrading aimlite" in captured or "v2.0.0" in captured
        assert "Successfully updated AIMLite" in captured or "2.1.1" in captured

    @patch("aimlite.cli.commands.update.get_installed_version")
    @patch("aimlite.cli.commands.update.get_latest_pypi_version", return_value="2.1.1")
    @patch("subprocess.run")
    def test_installs_when_not_installed(self, mock_run, mock_latest, mock_installed, capsys):
        mock_installed.side_effect = [None, "2.1.1"]
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        code = run_update()
        assert code == 0
        mock_run.assert_called()
        captured = capsys.readouterr().out
        assert "Installing aimlite" in captured
        assert "Successfully installed AIMLite" in captured

    @patch("aimlite.cli.commands.update.get_installed_version", return_value="2.1.1")
    @patch("aimlite.cli.commands.update.get_latest_pypi_version", return_value="2.1.1")
    @patch("subprocess.run")
    def test_force_reinstalls_even_if_current(self, mock_run, mock_latest, mock_installed, capsys):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        code = run_update(force=True)
        assert code == 0
        mock_run.assert_called()
        captured = capsys.readouterr().out
        assert "Reinstalling aimlite" in captured

    @patch("aimlite.cli.commands.update.get_installed_version", return_value="2.0.0")
    @patch("aimlite.cli.commands.update.get_latest_pypi_version", return_value="2.1.1")
    @patch("subprocess.run")
    def test_handles_installation_failure(self, mock_run, mock_latest, mock_installed, capsys):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "Permission denied"
        mock_proc.stdout = ""
        mock_run.return_value = mock_proc

        code = run_update()
        assert code == 1
        captured = capsys.readouterr().out
        assert "Failed to install/upgrade aimlite" in captured
