"""Tests for the GitHub Actions CI workflow (review 8.1).

The suite has been offline-green for months; CI converts that claim into
something a reader can verify on every push. The workflow file itself is
checked structurally here — the real run happens on GitHub."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"


class TestCiWorkflow:
    def test_workflow_file_exists(self) -> None:
        assert WORKFLOW.is_file()

    def test_matrix_covers_311_and_312_independently(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "windows-latest" in text
        assert '"3.11"' in text and '"3.12"' in text
        assert "fail-fast: false" in text  # 3.11 and 3.12 fail independently

    def test_workflow_installs_and_runs_pytest(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "pip install -e .[dev,gui]" in text
        assert "pytest" in text

    def test_triggers_on_push_and_pr(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "on:" in text
        assert "pull_request" in text
