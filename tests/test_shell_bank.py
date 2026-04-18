"""Tests for the formal shell-bank resource layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expert_data.shells import (
    SHELL_PLACEHOLDER_RULES,
    build_deterministic_shell_bank,
    extract_placeholders,
    validate_shell_bank,
)


class ShellBankTest(unittest.TestCase):
    """Validate subtype coverage and placeholders for the shell bank."""

    def test_shell_bank_subtype_counts(self) -> None:
        """Each subtype should meet the requested minimum template count."""

        shell_bank = build_deterministic_shell_bank()
        self.assertGreaterEqual(len(shell_bank["cat"]), 4)
        self.assertGreaterEqual(len(shell_bank["cnt"]), 3)
        self.assertGreaterEqual(len(shell_bank["col"]), 3)
        self.assertGreaterEqual(len(shell_bank["rel"]), 3)

    def test_shell_bank_placeholders_are_valid(self) -> None:
        """Every shell template should use only the allowed subtype placeholders."""

        shell_bank = build_deterministic_shell_bank()
        validate_shell_bank(shell_bank)
        for subtype, entries in shell_bank.items():
            allowed = SHELL_PLACEHOLDER_RULES[subtype]
            for entry in entries:
                q_placeholders = extract_placeholders(entry["q_template"])
                r_placeholders = extract_placeholders(entry["r_template"])
                combined = q_placeholders | r_placeholders
                self.assertFalse(
                    combined - allowed,
                    msg=f"Subtype '{subtype}' uses illegal placeholders: {combined - allowed}",
                )
                self.assertFalse(
                    allowed - combined,
                    msg=f"Subtype '{subtype}' misses required placeholders: {allowed - combined}",
                )


if __name__ == "__main__":
    unittest.main()
