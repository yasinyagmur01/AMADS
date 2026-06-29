"""Unit tests for deterministic Referee metric helpers (no LLM)."""

from __future__ import annotations

import unittest

from core.state import AgentDecision, ShockEvent, ShockType
from referee.referee_node import (
    _apply_pool_formula,
    _apply_shocks,
    _cooperation_score_avg,
    _gini_coefficient,
    _is_collapsed,
)


class GiniCoefficientTests(unittest.TestCase):
    def test_empty_returns_zero(self) -> None:
        self.assertEqual(_gini_coefficient([]), 0.0)

    def test_all_zero_extractions(self) -> None:
        self.assertEqual(_gini_coefficient([0.0, 0.0, 0.0]), 0.0)

    def test_perfect_equality(self) -> None:
        self.assertAlmostEqual(_gini_coefficient([5.0, 5.0, 5.0]), 0.0, places=6)

    def test_maximum_inequality_two_agents(self) -> None:
        self.assertAlmostEqual(_gini_coefficient([0.0, 10.0]), 0.5, places=6)

    def test_clamped_to_unit_interval(self) -> None:
        gini = _gini_coefficient([1.0, 100.0, 200.0])
        self.assertGreaterEqual(gini, 0.0)
        self.assertLessEqual(gini, 1.0)


class CooperationScoreTests(unittest.TestCase):
    def test_empty_extractions(self) -> None:
        self.assertEqual(_cooperation_score_avg([], 12.0), 0.0)

    def test_zero_max_extractable_full_cooperation(self) -> None:
        self.assertEqual(_cooperation_score_avg([0.0, 0.0], 0.0), 1.0)

    def test_full_extraction_zero_cooperation(self) -> None:
        self.assertAlmostEqual(
            _cooperation_score_avg([12.0, 12.0], 12.0), 0.0, places=6
        )

    def test_half_extraction(self) -> None:
        self.assertAlmostEqual(
            _cooperation_score_avg([6.0, 6.0], 12.0), 0.5, places=6
        )

    def test_clamped_per_agent(self) -> None:
        score = _cooperation_score_avg([20.0, 0.0], 10.0)
        self.assertAlmostEqual(score, 0.5, places=6)


class PoolFormulaTests(unittest.TestCase):
    def test_basic_depletion_and_regen(self) -> None:
        # pool=100, extract 30 → 70 clamped → ×1.15 = 80.5
        result = _apply_pool_formula(100.0, 100.0, 1.15, 30.0)
        self.assertAlmostEqual(result, 80.5, places=6)

    def test_clamp_at_zero_before_regen(self) -> None:
        result = _apply_pool_formula(50.0, 100.0, 1.15, 80.0)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_clamp_at_capacity(self) -> None:
        result = _apply_pool_formula(100.0, 100.0, 1.0, 0.0)
        self.assertAlmostEqual(result, 100.0, places=6)


class CollapseEpsilonTests(unittest.TestCase):
    def test_not_collapsed_above_epsilon(self) -> None:
        # COLLAPSE_EPSILON_RATIO default 0.01 → epsilon = 1.0 at cap 100
        self.assertFalse(_is_collapsed(1.5, 100.0))

    def test_collapsed_at_epsilon(self) -> None:
        self.assertTrue(_is_collapsed(1.0, 100.0))

    def test_collapsed_below_epsilon(self) -> None:
        self.assertTrue(_is_collapsed(0.0001, 100.0))

    def test_functional_death_not_exact_zero(self) -> None:
        """Asymptotic pool residue must still count as collapse."""
        self.assertFalse(_is_collapsed(1.5, 100.0))
        self.assertTrue(_is_collapsed(0.99, 100.0))


class ApplyShocksTests(unittest.TestCase):
    def test_capacity_drop_scales_pool(self) -> None:
        shocks = [
            ShockEvent(
                round_number=7,
                shock_type=ShockType.CAPACITY_DROP,
                magnitude=-0.20,
                seed_source="test",
            )
        ]
        cap, regen, pool, mult = _apply_shocks(
            100.0, 1.15, 80.0, shocks, round_number=7
        )
        self.assertAlmostEqual(cap, 80.0, places=6)
        self.assertAlmostEqual(pool, 64.0, places=6)
        self.assertAlmostEqual(regen, 1.15, places=6)
        self.assertAlmostEqual(mult, 1.0, places=6)

    def test_no_shock_when_round_mismatch(self) -> None:
        shocks = [
            ShockEvent(
                round_number=7,
                shock_type=ShockType.CAPACITY_DROP,
                magnitude=-0.20,
                seed_source="test",
            )
        ]
        cap, regen, pool, mult = _apply_shocks(
            100.0, 1.15, 80.0, shocks, round_number=6
        )
        self.assertAlmostEqual(cap, 100.0, places=6)
        self.assertAlmostEqual(pool, 80.0, places=6)

    def test_demand_surge_multiplier(self) -> None:
        shocks = [
            ShockEvent(
                round_number=0,
                shock_type=ShockType.DEMAND_SURGE,
                magnitude=0.10,
                seed_source="test",
            )
        ]
        _, _, _, mult = _apply_shocks(100.0, 1.15, 100.0, shocks, round_number=0)
        self.assertAlmostEqual(mult, 1.10, places=6)


class ConstraintViolationInMetrics(unittest.TestCase):
    def test_violation_counted_in_snapshot_path(self) -> None:
        from referee.referee_node import _compute_metrics_snapshot

        decisions = [
            AgentDecision(
                agent_id="a1",
                round_number=0,
                extraction_amount=13.0,
                justification="x",
                declared_max=12.0,
            ),
            AgentDecision(
                agent_id="a2",
                round_number=0,
                extraction_amount=6.0,
                justification="y",
                declared_max=12.0,
            ),
        ]
        snap = _compute_metrics_snapshot(0, decisions, 12.0, 90.0)
        self.assertEqual(snap.constraint_violations, 1)
        self.assertAlmostEqual(snap.total_extraction, 19.0, places=6)


if __name__ == "__main__":
    unittest.main()
