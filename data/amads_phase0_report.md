# AMADS Phase 0 — Veri Özeti

Oluşturulma: 2026-06-29
Kaynak DB: `data/results.db`
Round penceresi: **round 0** (confound'suz trait fidelity analizi)
Başlangıç havuz kapasitesi: **100.0**

## Deney envanteri

- **full_experiment_v1**: 45 run, 576 metrics satırı, 2880 agent_decisions satırı
- **control_group_v1**: 27 run, 351 metrics satırı, 1755 agent_decisions satırı

## Round-0 özet (deney bazında)

### full_experiment_v1 (n=45)

| Metrik | Ortalama | SD |
|---|---|---|
| extraction_fraction | 0.6519 | 0.1430 |
| cooperation_score_r0 | 0.3506 | 0.1445 |
| gini_r0 | 0.0076 | 0.0139 |
| total_extraction / pool_capacity | 0.3896 | 0.0867 |
| rounds_played (tam run) | 12.80 | — |
| erken collapse (round < 15) | 28/45 | — |

**Trait fidelity (round 0, Pearson r):**
- cooperation → extraction_fraction: r = 0.4563
- risk_tolerance → extraction_fraction: r = 0.6783

### control_group_v1 (n=27)

| Metrik | Ortalama | SD |
|---|---|---|
| extraction_fraction | 0.5000 | 0.2496 |
| cooperation_score_r0 | 0.5000 | 0.2496 |
| gini_r0 | 0.0000 | 0.0000 |
| total_extraction / pool_capacity | 0.3000 | 0.1498 |
| rounds_played (tam run) | 13.00 | — |
| erken collapse (round < 15) | 9/27 | — |

**Trait fidelity (round 0, Pearson r):**
- cooperation → extraction_fraction: r = -1.0000
- risk_tolerance → extraction_fraction: r = 0.0000

## Maliyet kalibrasyonu (full_experiment_v1)

| Kaynak | Değer |
|---|---|
| Model | `claude-haiku-4-5-20251001` |
| Logged toplam maliyet | $6.2728 |
| Run başına (45 run) | $0.1394 |
| Master ref tahmini (Haiku tam run) | ~$0.17 |
| Master ref tahmini (Sonnet tam run) | ~$0.50 |

## Export CSV sütunları

`amads_round0_summary.csv` alanları:

- `extraction_fraction_r0` — ortalama(extraction / declared_max), round 0
- `cooperation_score_r0`, `gini_r0` — Referee round-0 metrikleri
- `extraction_over_capacity_r0` — total_extraction_r0 / başlangıç kapasitesi
- `collapse_round`, `terminated_early` — erken sonlanma (round < max_rounds)

## Çıktı dosyaları

- `data/amads_round0_summary.csv` — run başına round-0 satırları
- `data/amads_phase0_report.md` — bu rapor
