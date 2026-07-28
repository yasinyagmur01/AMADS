"""Unit tests for the LLM-free Stag Hunt Referee (payoff matrix + metrics)."""

from __future__ import annotations

import unittest

from scenarios.stag_hunt.referee import (
    compute_coordination_rate,
    compute_early_late_stag_rates,
    compute_payoffs,
    compute_round_metrics,
    compute_stag_rate,
    parse_choice,
)
from scenarios.stag_hunt.state import (
    AGENT_A_ID,
    AGENT_B_ID,
    HARE_HARE,
    HARE_WHEN_PARTNER_STAGS,
    STAG_STAG,
    STAG_WHEN_PARTNER_HARES,
)


class ParseChoiceTests(unittest.TestCase):
    def test_parses_stag_and_hare(self) -> None:
        self.assertEqual(parse_choice("stag"), "stag")
        self.assertEqual(parse_choice("hare"), "hare")

    def test_case_insensitive_and_whitespace(self) -> None:
        self.assertEqual(parse_choice(" Stag "), "stag")
        self.assertEqual(parse_choice("HARE"), "hare")

    def test_unrecognized_defaults_to_hare(self) -> None:
        self.assertEqual(parse_choice("rabbit"), "hare")
        self.assertEqual(parse_choice(""), "hare")

    def test_parses_dict(self) -> None:
        self.assertEqual(parse_choice({"choice": "stag"}), "stag")
        self.assertEqual(parse_choice({}), "hare")


class PayoffMatrixTests(unittest.TestCase):
    def test_mutual_stag_is_best(self) -> None:
        self.assertEqual(compute_payoffs("stag", "stag"), (STAG_STAG, STAG_STAG))

    def test_mutual_hare_is_safe(self) -> None:
        self.assertEqual(compute_payoffs("hare", "hare"), (HARE_HARE, HARE_HARE))

    def test_hare_vs_stag_mismatch(self) -> None:
        # own=hare, opponent=stag -> hare gets 4, stag gets 0
        self.assertEqual(
            compute_payoffs("hare", "stag"),
            (HARE_WHEN_PARTNER_STAGS, STAG_WHEN_PARTNER_HARES),
        )

    def test_stag_vs_hare_mismatch(self) -> None:
        # own=stag, opponent=hare -> stag gets 0, hare gets 4
        self.assertEqual(
            compute_payoffs("stag", "hare"),
            (STAG_WHEN_PARTNER_HARES, HARE_WHEN_PARTNER_STAGS),
        )

    def test_payoff_dominance_and_risk_dominance(self) -> None:
        # Payoff-dominant: mutual stag > mutual hare.
        self.assertGreater(STAG_STAG, HARE_HARE)
        # Risk-dominant: hare is the safe choice — playing hare never
        # yields the worst outcome, while playing stag can yield zero.
        self.assertGreater(HARE_WHEN_PARTNER_STAGS, STAG_WHEN_PARTNER_HARES)
        self.assertEqual(STAG_WHEN_PARTNER_HARES, 0.0)


class RoundMetricsTests(unittest.TestCase):
    def test_cumulative_score_accumulates(self) -> None:
        m = compute_round_metrics(
            round_number=2,
            choice_a="stag",
            choice_b="stag",
            running_score_a=10.0,
            running_score_b=5.0,
        )
        self.assertAlmostEqual(m.agent_a_score, 10.0 + STAG_STAG)
        self.assertAlmostEqual(m.agent_b_score, 5.0 + STAG_STAG)
        self.assertTrue(m.mutual_stag)
        self.assertFalse(m.mutual_hare)
        self.assertFalse(m.miscoordinated)

    def test_mutual_hare_flag(self) -> None:
        m = compute_round_metrics(
            round_number=0,
            choice_a="hare",
            choice_b="hare",
            running_score_a=0.0,
            running_score_b=0.0,
        )
        self.assertTrue(m.mutual_hare)
        self.assertFalse(m.mutual_stag)
        self.assertFalse(m.miscoordinated)

    def test_miscoordination_flag(self) -> None:
        m = compute_round_metrics(
            round_number=0,
            choice_a="stag",
            choice_b="hare",
            running_score_a=0.0,
            running_score_b=0.0,
        )
        self.assertTrue(m.miscoordinated)
        self.assertFalse(m.mutual_stag)
        self.assertFalse(m.mutual_hare)
        self.assertEqual(m.agent_a_payoff, STAG_WHEN_PARTNER_HARES)
        self.assertEqual(m.agent_b_payoff, HARE_WHEN_PARTNER_STAGS)


class StagRateTests(unittest.TestCase):
    def _round(self, n: int, a: str, b: str) -> dict:
        return {"round_number": n, "agent_a_choice": a, "agent_b_choice": b}

    def test_all_stag_rate_one(self) -> None:
        history = [self._round(i, "stag", "stag") for i in range(5)]
        self.assertEqual(compute_stag_rate(history, AGENT_A_ID), 1.0)
        self.assertEqual(compute_stag_rate(history, AGENT_B_ID), 1.0)

    def test_mixed_stag_rate(self) -> None:
        history = [
            self._round(0, "stag", "hare"),
            self._round(1, "hare", "hare"),
        ]
        self.assertEqual(compute_stag_rate(history, AGENT_A_ID), 0.5)
        self.assertEqual(compute_stag_rate(history, AGENT_B_ID), 0.0)

    def test_empty_history_returns_zero(self) -> None:
        self.assertEqual(compute_stag_rate([], AGENT_A_ID), 0.0)


class CoordinationRateTests(unittest.TestCase):
    def _round(self, n: int, a: str, b: str) -> dict:
        return {"round_number": n, "agent_a_choice": a, "agent_b_choice": b}

    def test_full_coordination(self) -> None:
        history = [self._round(0, "stag", "stag"), self._round(1, "hare", "hare")]
        self.assertEqual(compute_coordination_rate(history), 1.0)

    def test_no_coordination(self) -> None:
        history = [self._round(0, "stag", "hare"), self._round(1, "hare", "stag")]
        self.assertEqual(compute_coordination_rate(history), 0.0)

    def test_partial_coordination(self) -> None:
        history = [
            self._round(0, "stag", "stag"),
            self._round(1, "stag", "hare"),
        ]
        self.assertEqual(compute_coordination_rate(history), 0.5)

    def test_empty_history_returns_zero(self) -> None:
        self.assertEqual(compute_coordination_rate([]), 0.0)


class EarlyLateStagRateTests(unittest.TestCase):
    def _round(self, n: int, a: str, b: str) -> dict:
        return {"round_number": n, "agent_a_choice": a, "agent_b_choice": b}

    def test_stag_early_hare_late(self) -> None:
        history = [self._round(i, "stag", "stag") for i in range(6)] + [
            self._round(i, "hare", "hare") for i in range(6, 12)
        ]
        early, late = compute_early_late_stag_rates(history, AGENT_A_ID, 12)
        self.assertEqual(early, 1.0)
        self.assertEqual(late, 0.0)

    def test_empty_history_returns_zero(self) -> None:
        early, late = compute_early_late_stag_rates([], AGENT_A_ID, 12)
        self.assertEqual(early, 0.0)
        self.assertEqual(late, 0.0)


if __name__ == "__main__":
    unittest.main()
