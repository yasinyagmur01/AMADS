import asyncio

from core.config import settings
from core.graph import app
from core.state import SimulationState, EnvironmentSnapshot, TraitProfile

# Başlangıç state'ini (yakıtı) hazırlıyoruz
initial_state = SimulationState(
    experiment_id="test_001",
    run_id="run_001",
    max_rounds=3, # Sadece 3 round test edelim, hızlı bitsin
    agent_traits={
        "agent_1": TraitProfile(agent_id="agent_1", cooperation_assigned=0.8, risk_tolerance_assigned=0.2, profile_label="Cooperative"),
        "agent_2": TraitProfile(agent_id="agent_2", cooperation_assigned=0.2, risk_tolerance_assigned=0.8, profile_label="Selfish")
    },
    shock_schedule=[],
    environment=EnvironmentSnapshot(
        pool_current=100.0,
        pool_capacity=100.0,
        regen_rate=1.1,
        max_extractable_this_round=20.0,
        round_number=0,
        is_collapsed=False
    )
)

print("🚀 Motor ateşleniyor...")

# LangGraph'ı invoke ederek çalıştırıyoruz
final_state = asyncio.run(app.ainvoke(initial_state))

print("\n--- 🧠 AJAN KARARLARI VE GEREKÇELERİ ---")
for decision in final_state['round_decisions']:
    print(f"Agent: {decision.agent_id} | Çekim: {decision.extraction_amount} | Gerekçe: {decision.justification}")

print("---")
print("🏁 Simülasyon bitti!")
print(f"Ulaşılan Round: {final_state['round_number']}")
# Köşeli parantez yerine nokta kullandık:
print(f"Kalan Havuz: {final_state['environment'].pool_current:.2f}")
print(f"Terminasyon Sebebi: {final_state['termination_reason']}")