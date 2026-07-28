"""Unit tests for the LLM-free iterated PD Referee (payoff matrix + metrics)."""

from __future__ import annotations

import unittest

from scenarios.iterated_pd.referee import (
    compute_early_late_coop_rates,
    compute_forgiveness_rate,
    compute_payoffs,
    compute_round_metrics,
    parse_choice,
)
from scenarios.iterated_pd.state import (
    AGENT_A_ID,
    AGENT_B_ID,
    P_PUNISHMENT,
    R_REWARD,
    S_SUCKER,
    T_TEMPTATION,
)


class ParseChoiceTests(unittest.TestCase):
    def test_parses_c_and_d(self) -> None:
        self.assertEqual(parse_choice("C"), "C")
        self.assertEqual(parse_choice("D"), "D")

    def test_case_insensitive_and_whitespace(self) -> None:
        self.assertEqual(parse_choice(" c "), "C")
        self.assertEqual(parse_choice("d"), "D")

    def test_unrecognized_defaults_to_defect(self) -> None:
        self.assertEqual(parse_choice("cooperate"), "D")
        self.assertEqual(parse_choice(""), "D")

    def test_parses_dict(self) -> None:
        self.assertEqual(parse_choice({"choice": "C"}), "C")
        self.assertEqual(parse_choice({}), "D")


class PayoffMatrixTests(unittest.TestCase):
    def test_mutual_cooperation_is_reward(self) -> None:
        self.assertEqual(compute_payoffs("C", "C"), (R_REWARD, R_REWARD))

    def test_mutual_defection_is_punishment(self) -> None:
        self.assertEqual(compute_payoffs("D", "D"), (P_PUNISHMENT, P_PUNISHMENT))

    def test_defect_against_cooperate_is_temptation_sucker(self) -> None:
        self.assertEqual(compute_payoffs("D", "C"), (T_TEMPTATION, S_SUCKER))

    def test_cooperate_against_defect_is_sucker_temptation(self) -> None:
        self.assertEqual(compute_payoffs("C", "D"), (S_SUCKER, T_TEMPTATION))

    def test_standard_pd_ordering_holds(self) -> None:
        self.assertGreater(T_TEMPTATION, R_REWARD)
        self.assertGreater(R_REWARD, P_PUNISHMENT)
        self.assertGreater(P_PUNISHMENT, S_SUCKER)
        # 2R > T + S: mutual cooperation beats alternating exploitation
        self.assertGreater(2 * R_REWARD, T_TEMPTATION + S_SUCKER)


class RoundMetricsTests(unittest.TestCase):
    def test_cumulative_score_accumulates(self) -> None:
        m = compute_round_metrics(
            round_number=3,
            choice_a="C",
            choice_b="C",
            running_score_a=6.0,
            running_score_b=9.0,
        )
        self.assertAlmostEqual(m.agent_a_score, 6.0 + R_REWARD)
        self.assertAlmostEqual(m.agent_b_score, 9.0 + R_REWARD)
        self.assertTrue(m.mutual_cooperation)
        self.assertFalse(m.mutual_defection)

    def test_mutual_defection_flag(self) -> None:
        m = compute_round_metrics(
            round_number=0,
            choice_a="D",
            choice_b="D",
            running_score_a=0.0,
            running_score_b=0.0,
        )
        self.assertTrue(m.mutual_defection)
        self.assertFalse(m.mutual_cooperation)

    def test_mixed_choice_flags_both_false(self) -> None:
        m = compute_round_metrics(
            round_number=0,
            choice_a="D",
            choice_b="C",
            running_score_a=0.0,
            running_score_b=0.0,
        )
        self.assertFalse(m.mutual_cooperation)
        self.assertFalse(m.mutual_defection)
        self.assertEqual(m.agent_a_payoff, T_TEMPTATION)
        self.assertEqual(m.agent_b_payoff, S_SUCKER)


class ForgivenessRateTests(unittest.TestCase):
    def _round(self, n: int, a: str, b: str) -> dict:
        return {"round_number": n, "agent_a_choice": a, "agent_b_choice": b}

    def test_no_defection_against_returns_zero(self) -> None:
        history = [self._round(0, "C", "C"), self._round(1, "C", "C")]
        self.assertEqual(compute_forgiveness_rate(history, AGENT_A_ID), 0.0)

    def test_full_forgiveness(self) -> None:
        # B defects against A in round 0; A still cooperates in round 1.
        history = [self._round(0, "C", "D"), self._round(1, "C", "C")]
        self.assertEqual(compute_forgiveness_rate(history, AGENT_A_ID), 1.0)

    def test_no_forgiveness_retaliation(self) -> None:
        # B defects against A in round 0; A retaliates with D in round 1.
        history = [self._round(0, "C", "D"), self._round(1, "D", "D")]
        self.assertEqual(compute_forgiveness_rate(history, AGENT_A_ID), 0.0)

    def test_partial_forgiveness(self) -> None:
        history = [
            self._round(0, "C", "D"),
            self._round(1, "C", "C"),  # forgave
            self._round(2, "C", "D"),
            self._round(3, "D", "C"),  # did not forgive
        ]
        self.assertAlmostEqual(compute_forgiveness_rate(history, AGENT_A_ID), 0.5)

    def test_symmetric_for_agent_b(self) -> None:
        # A defects against B in round 0; B forgives in round 1.
        history = [self._round(0, "D", "C"), self._round(1, "C", "C")]
        self.assertEqual(compute_forgiveness_rate(history, AGENT_B_ID), 1.0)


class EarlyLateCoopRateTests(unittest.TestCase):
    def _round(self, n: int, a: str, b: str) -> dict:
        return {"round_number": n, "agent_a_choice": a, "agent_b_choice": b}

    def test_all_cooperate_both_halves(self) -> None:
        history = [self._round(i, "C", "C") for i in range(12)]
        early, late = compute_early_late_coop_rates(history, AGENT_A_ID, 12)
        self.assertEqual(early, 1.0)
        self.assertEqual(late, 1.0)

    def test_cooperate_early_defect_late(self) -> None:
        history = [self._round(i, "C", "C") for i in range(6)] + [
            self._round(i, "D", "D") for i in range(6, 12)
        ]
        early, late = compute_early_late_coop_rates(history, AGENT_A_ID, 12)
        self.assertEqual(early, 1.0)
        self.assertEqual(late, 0.0)

    def test_empty_history_returns_zero(self) -> None:
        early, late = compute_early_late_coop_rates([], AGENT_A_ID, 12)
        self.assertEqual(early, 0.0)
        self.assertEqual(late, 0.0)


if __name__ == "__main__":
    unittest.main()
