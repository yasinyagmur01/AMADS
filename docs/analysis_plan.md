# AMADS — Cross-Model Analysis Plan (prereg-lite)

**Durum:** Onaylı, 2026-06-29  
**Amaç:** Birincil bulguların yalnızca `claude-haiku-4-5` artefaktı olmadığını, en azından ikinci bir Claude modelinde yön olarak tekrarlanıp tekrarlanmadığını test etmek.

---

## Kilitli (yeniden analiz edilmez)

- `full_experiment_v1` (Haiku, 45 run) — trait fidelity ana bulguları
- `control_group_v1`, kümeleme, `data/synthesis_report.md`

---

## Confirmatory mikro replikasyon

| Parametre | Değer |
|---|---|
| `experiment_id` | `sonnet_crossmodel_v1` |
| Model | `claude-sonnet-4-6` (Sonnet 4.6) |
| Script | `analysis/prompt_ab_sonnet_crossmodel.py` |
| Tasarım | cooperation ∈ {0.2, 0.8} × risk_tolerance ∈ {0.2, 0.8} |
| Tekrar / hücre | n = 10 |
| Toplam LLM çağrısı | 40 (tek agent, tek round) |
| Prompt | Türkçe — `decision_agent._build_system_prompt` ile aynı metin |
| temperature | 0.2 |
| Maliyet tavanı | $2.00 |
| DB | `data/sonnet_crossmodel_v1.csv` (results.db'ye yazılmaz) |

---

## Hipotezler (yön testi, p iddiası yok)

| ID | Hipotez | Haiku referans (round 0) |
|---|---|---|
| H1 | cooperation_assigned ↑ → extraction_fraction ↑ (ters fidelity) | r ≈ +0.46 |
| H2 | risk_tolerance_assigned ↑ → extraction_fraction ↑ | r ≈ +0.68 |

**Başarı kriteri (yönsel):** Sonnet'te H1 ve H2 Pearson r işareti Haiku ile aynı (pozitif).

---

## Dışlanan metrikler

- Within-run Gini (homojen agent tasarımı)
- `cooperation_score_avg` literatür kıyası
- Tam 45-run factorial replikasyon (future work)

---

## Çıktı

- `data/sonnet_crossmodel_v1.csv`
- Terminal özeti: hücre ortalamaları, r değerleri, tahmini maliyet
