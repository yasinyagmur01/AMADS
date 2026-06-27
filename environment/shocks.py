from core.state import ShockEvent, ShockType


def build_mock_dev_shock_schedule() -> list[ShockEvent]:
    """Mock geliştirme koşusu için seedli şok takvimi (round ~7-8, Bölüm 3)."""
    return [
        ShockEvent(
            round_number=7,
            shock_type=ShockType.CAPACITY_DROP,
            magnitude=-0.20,
            seed_source="mock_dev",
        ),
    ]


if __name__ == "__main__":
    schedule = build_mock_dev_shock_schedule()
    shock = schedule[0]
    assert shock.round_number == 7
    assert shock.shock_type == ShockType.CAPACITY_DROP
    assert shock.magnitude == -0.20
    print(f"OK: round={shock.round_number}, type={shock.shock_type}, magnitude={shock.magnitude}")
