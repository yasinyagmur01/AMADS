"""Language-variant selection for CPR / bargaining prompts (no LLM)."""

from __future__ import annotations

import unittest

from agents.decision_agent import (
    _build_human_prompt,
    _build_system_prompt,
    uses_english_prompts,
)
from core.state import AgentInputView, EnvironmentSnapshot, TraitProfile
from scenarios.bargaining.agents import (
    _build_proposer_system,
    uses_english_prompts as bargaining_uses_english,
)
from scenarios.bargaining.state import BargainingProposerView


def _sample_cpr_view() -> AgentInputView:
    return AgentInputView(
        own_trait=TraitProfile(
            agent_id="agent_1",
            cooperation_assigned=0.5,
            risk_tolerance_assigned=0.5,
            profile_label="mid",
        ),
        environment=EnvironmentSnapshot(
            pool_current=100.0,
            pool_capacity=100.0,
            regen_rate=1.15,
            max_extractable_this_round=12.0,
            round_number=1,
            is_collapsed=False,
        ),
        round_number=1,
    )


def _sample_proposer_view(experiment_id: str) -> BargainingProposerView:
    return BargainingProposerView(
        experiment_id=experiment_id,
        own_trait=TraitProfile(
            agent_id="proposer",
            cooperation_assigned=0.5,
            risk_tolerance_assigned=0.5,
            profile_label="mid",
        ),
        env_snapshot=EnvironmentSnapshot(
            pool_current=100.0,
            pool_capacity=100.0,
            regen_rate=1.0,
            max_extractable_this_round=100.0,
            round_number=1,
            is_collapsed=False,
        ),
        time_pressure=0.3,
        resource_scarcity=0.3,
        round_number=1,
        pie_size=100.0,
        recent_history=[],
    )


class TestPromptLanguage(unittest.TestCase):
    def test_locked_cpr_ids_remain_turkish(self) -> None:
        for eid in (
            "full_experiment_v1",
            "control_group_v1",
            "heterogeneous_v1",
            "prompt_revision_v1",
        ):
            self.assertFalse(uses_english_prompts(eid), eid)
            sys_prompt = _build_system_prompt(_sample_cpr_view(), experiment_id=eid)
            self.assertIn("Sen bir", sys_prompt)
            self.assertNotIn("You are an agent", sys_prompt)

    def test_english_cpr_ids(self) -> None:
        for eid in ("full_experiment_en_v1", "full_experiment_groq_v1", "groq_smoke_test"):
            self.assertTrue(uses_english_prompts(eid), eid)
            sys_prompt = _build_system_prompt(_sample_cpr_view(), experiment_id=eid)
            self.assertIn("You are an agent", sys_prompt)
            self.assertIn("fully selfish", sys_prompt)
            human = _build_human_prompt(_sample_cpr_view(), experiment_id=eid)
            self.assertIn("Pool:", human)

    def test_prompt_revision_keeps_behavioral_turkish(self) -> None:
        sys_prompt = _build_system_prompt(
            _sample_cpr_view(), experiment_id="prompt_revision_v1"
        )
        self.assertIn("havuzdan mümkün olan maksimum", sys_prompt)

    def test_locked_bargaining_turkish(self) -> None:
        for eid in ("bargaining_v1", "bargaining_risk_v1"):
            self.assertFalse(bargaining_uses_english(eid), eid)
            prompt = _build_proposer_system(_sample_proposer_view(eid))
            self.assertIn("TEKLİF EDEN", prompt)

    def test_english_bargaining(self) -> None:
        eid = "bargaining_groq_v1"
        self.assertTrue(bargaining_uses_english(eid))
        prompt = _build_proposer_system(_sample_proposer_view(eid))
        self.assertIn("PROPOSER", prompt)
        self.assertIn("fully" if False else "aggressive offer", prompt)


if __name__ == "__main__":
    unittest.main()
