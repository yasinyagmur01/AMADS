# AMADS Analiz Sentez Raporu

## Veri kaynağı

- **LLM deneyi:** `full_experiment_v1` (45 run, experiment_conditions)
- **Kontrol grubu:** `control_group_v1` (27 run, experiment_conditions)
- **Round penceresi (fidelity / kontrol):** round 0–0
- **Kümeleme:** tüm round'lar, davranış vektörü (extraction_fraction, gini, cooperation_score_avg, collapse_round); k=2

## Özet tablo

| Yöntem | Risk'in etkisi | Cooperation'ın etkisi |
|---|---|---|
| Trait fidelity (LLM, r, p) | r=0.678, p<0.0001 | r=0.456, p=0.0016 (ters yön) |
| Kümeleme (k-means, trait'i hiç görmeden) | kümeleri ayırıyor (risk ekseni net) | kümeleri ayırmıyor |
| Kontrol grubu (deterministik formül) vs LLM yön karşılaştırması | (risk kontrol grubunda kullanılmadı, N/A) | kontrol: negatif (beklenen), LLM: pozitif (ters) |
| Cross-model replikasyon (Sonnet) | Haiku'nun tam tersi pattern | — |

## Sentez

Üç bağımsız yöntem tutarlı bir tablo çiziyor: risk_tolerance, hem Pearson korelasyonunda (r=0.678) hem de davranış kümelemesinde (ARI=0.286) LLM çıktısını güvenilir biçimde yönlendiriyor. cooperation_assigned ise ters yönde davranıyor — fidelity analizinde pozitif ve anlamlı bir ilişki (r=0.456, beklenen negatifin tersi), kümelemede ise koşulları ayırmıyor (ARI=0.123); deterministik kontrol grubu beklenen negatif yönü doğrularken LLM grubu pozitif yönde sapıyor. Bu bulgular, risk trait'inin modele aktarıldığını, cooperation trait'inin ise hem fidelity hem kümeleme düzeyinde etkisiz veya ters kaldığını gösteriyor. Claude Sonnet 4.6 ile yapılan cross-model replikasyonu (n=40), trait fidelity'nin model-spesifik olduğunu gösteriyor: Haiku'da ters olan cooperation, Sonnet'te beklenen yönde (r≈-0.84); Haiku'da güçlü olan risk ise Sonnet'te zayıfladı (r≈+0.15). Bu, 'concept misalignment' bulgusunun evrensel değil, model-bağımlı olduğuna işaret ediyor.

## Detay (referans)

### Trait fidelity (LLM)

- n = 45
- risk → extraction_fraction: r = 0.6783, p = 3.05e-07
- coop → extraction_fraction: r = 0.4563, p = 0.001631

### Kümeleme

- n = 45, k = 2
- ARI (küme ↔ risk_level): 0.2862
- ARI (küme ↔ coop_level): 0.1235

### Kontrol vs LLM (coop → extraction_fraction)

- Kontrol (27 run): r = -1.0000, p = 0
- LLM (45 run): r = 0.4563, p = 0.001631
