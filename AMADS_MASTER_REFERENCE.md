# AMADS — Master Architecture Reference
> Bu dosya, projenin tüm mimari kararlarının tek doğruluk kaynağıdır (single source of truth). Dokümantasyon yazarken veya Cursor'da kod yazdırırken buraya referans ver. Hiçbir karar burada yoksa, henüz resmi olarak verilmemiştir.

---

## 1. Proje Vizyonu

**Ad:** AMADS (Academic Multi-Agent Decision Simulation)

**Amaç:** LLM tabanlı agent'ların, farklı psikolojik trait profilleriyle (risk_tolerance, cooperation), kısıtlı bir çevresel senaryoda (Commons Dilemma) verdiği kararları **ölçülebilir, deterministik, döngüsel-olmayan** bir değerlendirme katmanıyla analiz etmek.

**Temel ilke:** "LLM'ler konuştu, bir şeyler oldu" değil — her sonuç kod tarafında hesaplanan, kanıtlanabilir bir sayıya dayanır. LLM çağrısı sadece **karar üretmek** için kullanılır, **karar değerlendirmek** için asla kullanılmaz.

**Kapsam-dışı (Scope):** Açık uçlu doğal dil pazarlığı, agent'lar arası direkt mesajlaşma, LLM-as-judge değerlendirme.

---

## 2. Senaryo: Commons Dilemma (Kaynak Paylaşımı)

### 2.1. Neden Bu Senaryo
- Oyun teorisinde 50+ yıllık literatür desteği (Hardin 1968, Ostrom)
- `cooperation` ve `risk_tolerance` trait'leri doğal olarak ve sayısal şekilde operasyonelleştirilebiliyor
- Round sayısı sabitlenebildiği için yakınsama garantisi var
- "Black Swan Events" (environment shock) roadmap'iyle doğal uyum

### 2.2. Havuz Dinamiği (Tam Deterministik, LLM'siz)
```
pool[t+1] = clamp(pool[t] - Σ(extraction_i[t]), 0, capacity) * regen_rate
```
- `capacity`: havuzun üst sınırı (sabit)
- `regen_rate`: yenilenme katsayısı (sabit, örn. 1.15)
- `pool == 0` → simülasyon "collapse" durumuna girer, erken sonlanma tetiklenir

### 2.3. Aksiyon Uzayı (Agent'ın Tek Çıktısı)
Her round'da her agent **yalnızca** şunu üretir (structured output / function calling ile):
- `extraction_amount: float` — havuzdan çekilen miktar
- `justification: str` — SADECE log amaçlı, hiçbir metrik hesaplamasına girmez
- `declared_max: float` — o round'da agent'a bildirilen üst sınırın kopyası (ihlal kontrolü için)

`max_extractable_this_round`, environment tarafından her round başında hesaplanır (örn. mevcut havuzun %40'ı) — agent bu sınırı göremez/değiştiremez, sadece içinde seçim yapar.

---

## 3. Sabit Parametreler (Pilot/Varsayılan Değerler)

| Parametre | Değer | Not |
|---|---|---|
| Agent sayısı | **5** | Literatürle uyumlu, kümeleme için yeterli çeşitlilik |
| Round sayısı | **15** | Şok öncesi/sonrası simetrik karşılaştırma alanı bırakır |
| Şok round'u | **~7-8** (kesin sayı henüz seçilmedi) | Açık detay — bkz. Bölüm 11 |
| Determinizm (temperature) | 0.2 | "Reproducibility garantisi" DEĞİL — stokastik varyans azaltma olarak çerçevelenir |
| Şok seed kaynağı | `experiment_id + run_id` | Aynı run_id her çalıştırıldığında aynı şokları üretir |

> Bu değerler config'te sabit değil, kolayca değiştirilebilir parametreler olarak tutulmalı (`core/config.py`).

---

## 4. State Şeması (Pydantic v2 — Tam Referans)

### Katman Mantığı (Kim Neyi Yazabilir)
```
SimulationState
├── environment: EnvironmentSnapshot       # SADECE Referee/Environment node yazar
├── agent_traits: Dict[str, TraitProfile]   # Sabit, asla değişmez (frozen)
├── round_decisions: List[AgentDecision]    # Agent node'ları YALNIZCA buraya EKLER
├── metrics_history: List[MetricsSnapshot]  # SADECE Referee yazar
├── shock_log: List[ShockEvent]             # Önceden hesaplanmış, frozen
└── round_number: int                       # SADECE Referee artırır
```

### 4.1. TraitProfile
```python
from pydantic import BaseModel, Field, ConfigDict

class TraitProfile(BaseModel):
    model_config = ConfigDict(frozen=True)
    agent_id: str
    cooperation_assigned: float = Field(ge=0.0, le=1.0)
    risk_tolerance_assigned: float = Field(ge=0.0, le=1.0)
    profile_label: str  # örn. "high_coop_low_risk"
```

### 4.2. EnvironmentSnapshot
```python
class EnvironmentSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    pool_current: float = Field(ge=0.0)
    pool_capacity: float = Field(gt=0.0)
    regen_rate: float = Field(gt=0.0)
    max_extractable_this_round: float = Field(ge=0.0)
    round_number: int = Field(ge=0)
    is_collapsed: bool = False
```

### 4.3. AgentDecision
```python
class AgentDecision(BaseModel):
    agent_id: str
    round_number: int
    extraction_amount: float = Field(ge=0.0)
    justification: str = Field(max_length=500)  # SADECE log
    declared_max: float = Field(ge=0.0)
```

### 4.4. ShockEvent
```python
from enum import Enum

class ShockType(str, Enum):
    CAPACITY_DROP = "capacity_drop"
    REGEN_BOOST = "regen_boost"
    DEMAND_SURGE = "demand_surge"

class ShockEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    round_number: int
    shock_type: ShockType
    magnitude: float  # örn. -0.30 -> %30 azalma
    seed_source: str  # audit için
```

### 4.5. MetricsSnapshot
```python
class MetricsSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    round_number: int
    gini_coefficient: float = Field(ge=0.0, le=1.0)
    cooperation_score_avg: float = Field(ge=0.0, le=1.0)
    total_extraction: float = Field(ge=0.0)
    pool_after: float = Field(ge=0.0)
    constraint_violations: int = Field(ge=0, default=0)
```

### 4.6. SimulationState (Ana State)
```python
from typing import Dict, List, Optional

class SimulationState(BaseModel):
    experiment_id: str
    run_id: str
    agent_traits: Dict[str, TraitProfile]
    shock_schedule: List[ShockEvent]
    max_rounds: int = Field(gt=0)

    environment: EnvironmentSnapshot
    round_decisions: List[AgentDecision] = []
    metrics_history: List[MetricsSnapshot] = []
    round_number: int = 0

    is_terminated: bool = False
    termination_reason: Optional[str] = None
```

### 4.7. Agent'ın Gördüğü Salt-Okunur Görünüm (Read-Only Enforcement)
```python
class AgentInputView(BaseModel):
    """Agent node'una geçirilen, environment'ı DEĞİŞTİREMEYECEĞİ salt-okunur görünüm"""
    model_config = ConfigDict(frozen=True)
    own_trait: TraitProfile
    environment: EnvironmentSnapshot
    round_number: int
```
> Read-only environment kısıtı, prompt seviyesi rica DEĞİL, **fonksiyon imzası seviyesi garanti**. Agent node fonksiyonu `SimulationState`'in tamamını değil, sadece `AgentInputView`'ı görür.

---

## 5. Trait → Davranış Köprüsü (Operationalization)

| Trait | Prompt'a nasıl giriyor | Ölçüm (post-hoc, koddan, LLM'siz) |
|---|---|---|
| cooperation (0-1) | Sistem promptunda davranışsal eğilim tonu | `1 - (agent_extraction / max_extractable)` ortalaması |
| risk_tolerance (0-1) | Sistem promptunda davranışsal eğilim tonu | Şok sonrası round'larda çekim davranışı varyansı |

> Kritik nokta: trait sadece **prompt girdisi**; gözlemlenen davranış ise agent'ın sayısal çıktısından **bağımsız olarak** hesaplanır. İkisi arası korelasyon (`Trait-Behavior Fidelity`) projenin asıl bulgularından biri.

**Açık detay:** Trait sayısının agent'a gönderilecek gerçek sistem promptu cümlesi henüz yazılmadı (bkz. Bölüm 11).

---

## 6. Metrikler (Tam Formel, LLM'siz)

| Metrik | Formül / Tanım |
|---|---|
| Sustainability Index | `collapse_round / total_rounds` (1 = hiç çökmedi) |
| Gini Katsayısı | standart Gini formülü, `extraction` dizisi üzerinden |
| Cooperation Score | round bazlı ortalama (Bölüm 5'teki formül) |
| Shock Resilience | şok öncesi/sonrası `pool` toparlanma hızı (round sayısı) |
| Trait-Behavior Fidelity | `corr(assigned_trait, observed_behavior)` — Pearson korelasyon |
| Constraint Violations | `extraction_amount > declared_max` sayısı |

---

## 7. Referee Node — Rolü

Referee **yorumlamaz, kayıt tutar ve kuralı uygular**:
1. Round'daki tüm `AgentDecision`'ları toplar
2. Havuz formülünü uygular (Bölüm 2.2)
3. Şok takvimini kontrol eder, varsa uygular
4. Metrikleri hesaplar (Bölüm 6)
5. `round_number`'ı artırır, terminasyon koşulunu kontrol eder

**Referee hiçbir LLM çağrısı yapmaz.** Bu, projenin "circular evaluation yok" iddiasının temelidir.

---

## 8. LangGraph Mimarisi (Teknik Katman)

### 8.1. Topoloji: Paralel Fan-out + Reduce
```
                  ┌─→ Agent1 (izole, paralel) ─┐
Round Başlat  ───┼─→ Agent2 (izole, paralel) ─┼─→ Referee (topla, hesapla, ilerlet)
                  └─→ Agent3...Agent5 ─────────┘
```
- Her agent, **o round'da diğerlerini görmeden** paralel çalışır → gerçek simultaneous game kuralı
- Round ilerlemesi hiçbir LLM kararına bağlı değil, sabit sayaç (`round_number < max_rounds`) kontrolünde

### 8.2. Teknik Kararlar
| Karar | Seçim | Neden |
|---|---|---|
| Graf inşa stili | **StateGraph** (Functional API değil) | Her node ayrı görünür, LangSmith trace uyumu yüksek |
| Paralel fan-out yöntemi | **Statik paralel edge** (`Send()` API değil) | Agent sayısı sabit (5), dinamik dallanmaya gerek yok |
| State güncelleme (reducer) | `round_decisions: Annotated[list, operator.add]` | 5 agent paralel yazarken çakışma olmaz, her biri ekler |
| `environment`, `metrics_history`, `round_number` | Reducer YOK, tek kaynak (Referee) yazar | Zaten tek node yazıyor |
| Checkpointing backend | **SqliteSaver** | Zaten SQLite kararıyla uyumlu, `run_id` ↔ `thread_id` eşlemesi |

### 8.3. Conditional Edge Mantığı (LLM'siz Dallanma)
```
Referee → [pool == 0 ?] → Terminate (termination_reason="collapse")
        → [round_number >= max_rounds ?] → Terminate (termination_reason="completed")
        → [else] → Yeni Round (Agent fan-out'a geri dön)
```

### 8.4. Subgraph Önerisi (Her Agent İçin)
`decide → self_check_constraint → finalize` — agent'ın kendi `declared_max`'i aşıp aşmadığını kendi kendine kontrol ettiği ek bir adım (henüz detaylandırılmadı, bkz. Bölüm 11).

---

## 9. Uygulama Sırası (Cursor'da İzlenecek Adımlar)

1. **Mock node'larla iskelet** — gerçek LLM yerine sabit/rastgele `AgentDecision` üreten stub'larla grafı baştan sona test et (sıfır maliyet)
2. **Conditional edge doğrulama** — `pool==0` ve `round>=max_rounds` koşullarını mock veriyle test et
3. **Checkpointing test** — bir run'ı ortadan kesip `thread_id` ile devam ettirme senaryosu
4. **Tek agent'ı gerçek LLM'e bağla** — 5'ini birden değil, önce 1'ini (maliyet kontrolü)
5. **LangSmith tracing aç** — trace'lerle "circular evaluation yok" iddiasını kanıtla
6. **Kalan 4 agent'ı bağla, tam run dene**

> Bir adım çalışmadan ikinciye geçilmez — debug karmaşıklığını katmanlara bölmemek spagettinin asıl nedeni.

---

## 10. Maliyet ve Kaynak Yönetimi

### 10.1. İlkeler
1. **Dry-run/mock modu** zorunlu (yukarıdaki adım 1)
2. **Maliyet tahmin scripti** — büyük N öncesi token tahmini yapan ön-kontrol
3. **Kademeli model stratejisi** — pilot/debug: Haiku 4.5, istatistiksel final runs: Sonnet 4.6
4. **Hard cap'ler** — `max_rounds`, `max_agents`, `max_runs_per_batch` config seviyesinde sabit tavanlar

### 10.2. Güncel Fiyatlar (Haziran 2026, MTok başına)
| Model | Input | Output |
|---|---|---|
| Haiku 4.5 | $1.00 | $5.00 |
| Sonnet 4.6 | $3.00 | $15.00 |

### 10.3. Tek Run Tahmini (5 agent × 15 round = 75 çağrı, stateless agent'lar)
| | Sonnet 4.6 | Haiku 4.5 |
|---|---|---|
| Çağrı başı (~1200 input / ~200 output token) | ~$0.0066 | ~$0.0022 |
| **Tam run (75 çağrı)** | **~$0.50** | **~$0.17** |

> Referee hiç LLM çağırmadığı için maliyete katkısı sıfır. Tahminler ±%30-50 sapabilir, gerçek ölçümle kalibre edilmeli.

---

## 11. Açık Kalan Teknik Detaylar — Durum (Faz 2 sonrası güncellendi)

1. ~~Şokun tam round numarası~~ → **KİLİTLİ: round 7**, `CAPACITY_DROP` (-%20), tüm kalibrasyon ve `full_experiment_v1` koşularında doğrulandı (LangSmith trace + pool_after kırılması ile kanıtlandı).
2. ~~Trait sayısının agent'a gönderilecek gerçek sistem promptu cümlesi~~ → **KİLİTLİ**, bkz. Bölüm 18.2 (tam metin + bulgu).
3. Subgraph'ın `self_check_constraint` adımının tam mantığı → **hâlâ açık**, henüz ele alınmadı (constraint violation şu an sadece referee'de post-hoc sayılıyor, agent'ın kendi kendini kısıtlaması ayrı bir mekanizma olarak yazılmadı).
4. ~~Checkpoint tablosunun SQLite'ta tam şeması~~ → **KİLİTLİ ve TEST EDİLDİ**: LangGraph'ın kendi `SqliteSaver` checkpoint dosyası (ayrı dosya, `data/results.db`'den bağımsız). Checkpointing testi (round 5'te kesip `thread_id` ile devam ettirme) başarıyla doğrulandı — round 0'dan değil round 5'ten devam etti, `metrics_history` round 0-4 birebir korundu.

## 12. Bilinçli Olarak En Sona Bırakılanlar — Durum

1. ~~İstatistiksel deney tasarımı~~ → **KİLİTLİ ve UYGULANDI**, bkz. Bölüm 18.3 (tam faktöriyel tasarım, N, power analysis sonuçları).
2. **Test stratejisi** — unit/integration test kapsamı, hangi senaryolar mock'lanacak → hâlâ açık, formel bir test stratejisi dokümante edilmedi (ad-hoc doğrulama scriptleriyle ilerlendi, bkz. Bölüm 18.5 — yapısal denetim).

---

## 13. Ek Analiz Modülleri (Core'dan Ayrı, Maliyetsiz)

Bunlar core simülasyon bittikten sonra, **ekstra LLM çağrısı gerektirmeden** mevcut veriye uygulanır:

| Modül | Soru | Veri Kaynağı | Dosya |
|---|---|---|---|
| Kontrol Grubu Agent | LLM davranışı, sabit kurallı bot'tan farklı mı? | Aynı simülasyondaki ham extraction verisi (LLM çağrısı yapmayan bir `AgentDecision` üreticisi) | `agents/control_agent.py` |
| Kümeleme (Clustering) | Kaç farklı strateji arketipi kendiliğinden ortaya çıkıyor? | Davranış vektörleri, scikit-learn k-means | `analysis/clustering.py` |
| İnsan Verisiyle Kıyas | Bulunan örüntüler gerçek insan davranışına ne kadar yakın? | Literatürden çekilen Gini/cooperation referans değerleri | `analysis/human_baseline.py` |

> Bu üçü birbirinin girdisini kullanan kademeli bir anlatı oluşturur: önce "ne oldu" (kümeleme), sonra "rastgele mi anlamlı mı" (kontrol grubu), sonra "gerçek dünyaya benziyor mu" (insan kıyası).

---

## 14. Docker & Servis Mimarisi

```yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.app
    volumes:
      - ./data:/app/data          # SQLite kalıcı
    env_file:
      - .env                       # ANTHROPIC_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING=true
    command: python run_simulation.py

  analytics:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    volumes:
      - ./data:/app/data:ro        # SADECE okuma — analiz katmanı veriye yazamaz
    ports:
      - "8501:8501"
    depends_on:
      - app
```

| Servis | Durum |
|---|---|
| app | LangGraph runtime, tek container |
| analytics | Streamlit, read-only data erişimi |
| db (SQLite) | Ayrı container DEĞİL — dosya bazlı, volume ile paylaşılıyor |
| ChromaDB | Dahil değil — izolasyon riski çözülmeden eklenmeyecek |
| LangSmith | Container değil — Cloud SaaS, sadece `.env` üzerinden API key |

---

## 15. Klasör Yapısı (İskelet)

```
amads/
├── core/
│   ├── state.py          # Bölüm 4'teki tüm Pydantic şemaları
│   ├── graph.py           # LangGraph StateGraph kurulumu
│   └── config.py          # max_rounds, agent sayısı, hard cap'ler
├── agents/
│   ├── decision_agent.py  # Gerçek LLM agent
│   └── control_agent.py   # Kontrol grubu (LLM'siz, sabit kural)
├── referee/
│   └── referee_node.py    # Deterministik hesaplama, LLM çağrısı YOK
├── environment/
│   └── shocks.py           # Seedli şok takvimi üretici
├── analysis/
│   ├── clustering.py
│   ├── control_comparison.py
│   └── human_baseline.py
├── tests/
│   └── (mock/dry-run senaryoları)
├── docker-compose.yml
├── Dockerfile.app
├── Dockerfile.streamlit
├── CLAUDE.md               # Cursor için kural dosyası
└── README.md
```

---

## 16. Dokümantasyon Seti (17 Doküman, Üretilecek)

| # | Doküman | Grup |
|---|---|---|
| 1 | Project Charter / Vision Doc | Kuş bakışı |
| 2 | README.md | Kuş bakışı |
| 3 | Glossary | Kuş bakışı |
| 4 | System Architecture Document | Mimari |
| 5 | Architecture Decision Records (ADR) | Mimari |
| 6 | Data/State Schema Reference | Mimari |
| 7 | LangGraph Flow & Node Contract Doc | Çekirdek |
| 8 | Sequence/Lifecycle Doc | Çekirdek |
| 9 | Experimental Design Doc | Akademik (en son) |
| 10 | Metrics & Operationalization Reference | Akademik |
| 11 | Analysis Modules Doc | Akademik |
| 12 | Docker & Deployment Guide | Operasyon |
| 13 | Cost & Resource Management Doc | Operasyon |
| 14 | Testing Strategy Doc | Kalite |
| 15 | CLAUDE.md / Cursor Rules | Kalite |
| 16 | Contribution Guide | Katkı |
| 17 | Changelog | Katkı |

> Diyagram gereken yerler (4, 7, 8) görsel araçla üretilecek, metin kısımları bu referanstan türetilecek.

---

## 17. Temel Karar Geçmişi (Özet — "Neden Böyle" Hatırlatması)

| Karar | Alternatifler | Neden Bu Seçildi |
|---|---|---|
| Commons Dilemma senaryosu | Pazarlık, Kriz | Literatür desteği en güçlü, trait operasyonelleştirme en doğal |
| Structured output (sayısal karar) | Açık uçlu doğal dil | Döngüsel değerlendirmeyi engelliyor, kod tarafında objektif okunabiliyor |
| Paralel fan-out | Sıralı sohbet, Supervisor | Gerçek simultaneous game, convergence garantisi, tekrarlanabilirlik |
| Deterministik Referee | LLM-as-judge | "Subjective değil quantitative" temel vaadini gerçek kılıyor |
| SQLite | Postgres | Tek kullanıcı/tek makine için yeterli, basit |
| StateGraph | Functional API | Şeffaflık, LangSmith trace uyumu |
| Statik paralel edge | `Send()` API | Agent sayısı sabit, gereksiz esneklik yok |
| Temperature = 0.2 (sabit, değiştirilmedi) | 0.35-0.5 ("kontrollü stokastiklik") | Gerçek varyans verisi (gini σ²≈4.78e-4) zaten yeterli sinyal verdi, determinizm ilkesini zayıflatmaya gerek kalmadı (Bölüm 18.3) |
| EXTRACTION_LIMIT_RATIO = 0.12 | 0.40 (ilk varsayım), 0.15, 0.30 | Mock + gerçek LLM kalibrasyonuyla seçildi: round 7 şokunu her zaman gören, ama 15'e hiç tamamlamayan dengeli nokta (Bölüm 18.1) |
| COLLAPSE_EPSILON_RATIO = 0.01 (yeni parametre) | `pool == 0` tam eşitlik (eski, hatalı) | Float'lar asimptotik olarak sıfıra yaklaşıp asla tam sıfır olmuyordu; "fonksiyonel ölüm" yanlış şekilde "completed" sayılıyordu (Bölüm 18.1) |
| Groq tamamen kaldırıldı | Groq + Anthropic hibrit | Mimaride hiç yoktu, model karışıklığı confound riski taşıyordu, maliyet kazancı ihmal edilebilirdi |

---

## 18. Faz 2 — Kalibrasyon, İlk Deney ve Bulgular (Log)

> Bu bölüm, mimari tasarımdan gerçek veri toplamaya geçişin tam günlüğüdür. Bölüm 1-17 "ne inşa edeceğiz" sorusuna cevaptı; bu bölüm "inşa ettik, çalıştırdık, ne öğrendik" sorusuna cevap.

### 18.1. Kalibrasyon Süreci ve Çözülen Bug'lar

**Epsilon bug'ı (kritik, veri geçerliliğini etkiliyordu):**
Referee'nin collapse kontrolü eskiden `pool_after <= 0` (tam matematiksel sıfır) arıyordu. Pool formülü (`clamp` + `regen_rate`) doğası gereği asimptotik küçülüyor, asla tam sıfıra ulaşmıyor (örn. 0.0000003 gibi bir kalıntıda kalıp sonsuza kadar "hayatta" sayılıyordu). Sonuç: bazı run'lar fonksiyonel olarak ölü olduğu halde (round 3'ten sonra hiçbir agent hiçbir şey çekemiyor) `termination_reason="completed"` ile 15 round'a "tamamlanmış" görünüyordu — bu, Sustainability Index ve Cooperation Score'u sistematik olarak bozacak bir ölçüm hatasıydı.

*Düzeltme:* `COLLAPSE_EPSILON_RATIO=0.01` eklendi → `is_collapsed = pool_after <= pool_capacity * 0.01`. Düzeltme sonrası aynı run'lar doğru şekilde round 1-2'de collapse olarak işaretlendi.

**`_apply_shocks` hiç çalışmıyordu (bug):** Şok (CAPACITY_DROP, round 7, -%20) pool'a hiçbir etki yaratmıyordu — round'lar arası pool oranı tamamen sabitti. Düzeltme: şok artık `pool_capacity` ve `pool_current`'ı orantılı küçültüyor (`scale = 1 + magnitude`). Doğrulama: round 6→7 arası oran 0.6325→0.5060'a düştü (≈%20 ek düşüş, beklenen).

**EXTRACTION_LIMIT_RATIO kalibrasyonu (mock → gerçek LLM):**

| Ratio | Test türü | R7 (şoku görme oranı) | Done15 (tamamlama) | Karar |
|---|---|---|---|---|
| 0.40 | Mock + gerçek LLM | %0 | %0 | Çok agresif, elendi |
| 0.30 | Mock (n=40) | %37.5 | %22.5 | Mock'ta umut verici |
| 0.30 | **Gerçek LLM (n=3)** | %0 (epsilon fix sonrası) | %0 | Mock'tan çok daha agresif — gerçek LLM'ler mock'tan farklı davranıyor |
| 0.15 | Gerçek LLM (n=3) | %100 | %0 | Hep şoku görüyor, hep collapse round 8-9 |
| **0.12** | **Gerçek LLM (n=3)** | **%100** | **%0 (her zaman round 12'de collapse)** | **KİLİTLİ** — şok her zaman test edilebiliyor |

> Önemli not: mock kalibrasyonu ile gerçek LLM davranışı **uyuşmadı** (mock'ta 0.30 "iyi" görünürken gerçek LLM'de aynı ratio %0 R7 verdi). Bu, mock'un rastgele extraction'ının (`uniform(0.05, 0.95)`) gerçek LLM'in trait-yönlendirmeli, daha tutarlı davranışından yapısal olarak farklı olduğunu gösteriyor. **Sonuç: ileride parametre kalibrasyonu her zaman gerçek LLM ile son onaylanmalı, mock sadece kaba bir ön-tarama.**

### 18.2. Trait → Prompt Operasyonelleştirmesi (KİLİTLİ)

`agents/decision_agent.py`, `_build_system_prompt()`:

```
"Sen bir ortak kaynak havuzundan çekim yapan bir agentsın.
cooperation_assigned değerin {cooperation_assigned:.2f}
(0=tamamen bencil, 1=tamamen işbirlikçi), risk_tolerance_assigned
değerin {risk_tolerance_assigned:.2f} (0=çok temkinli, 1=çok risk alan).
Kararını bu eğilimlere uygun ver, ama bu sayıları çıktında tekrar
etme veya açıklama."
```

Bu cümle son commit ile doğrulandı (git diff boş), coop ve risk ifadeleri yazım olarak simetrik — bu, Bölüm 18.4'teki bulgunun bir prompt asimetrisi olmadığını kanıtlamak için önemli bir referans noktası.

### 18.3. İstatistiksel Deney Tasarımı (KİLİTLİ, `full_experiment_v1`)

| Parametre | Değer |
|---|---|
| Trait sayısı | 2 (`cooperation_assigned`, `risk_tolerance_assigned`) |
| Seviye/trait | 3 (Low=0.2, Medium=0.5, High=0.8) |
| Koşul sayısı | 3×3 = 9 (tam faktöriyel, her koşulda 5 agent'a aynı trait çifti) |
| Tekrar/koşul (N) | 5 (toplam 45 run) |
| max_rounds | 15 |
| Model | claude-haiku-4-5-20251001 (pilot/kalibrasyon aşaması; final istatistiksel run'lar için Sonnet 4.6 planlanıyor) |
| Sonuç | 45/45 run tamamlandı, $6.27 maliyet, hepsi `collapse` (round 10-15 arası), `experiment_conditions` tablosu ile run_id↔trait eşlemesi DB'de kayıtlı |

**Power analysis bulgusu (önemli, raporlamada dürüst kalınmalı):**
Aynı mutlak fark eşiği (δ=0.05), farklı metriklerde farklı istatistiksel güç veriyor — çünkü her metriğin kendi doğal varyansı farklı:
- `gini_coefficient`: σ≈0.0197 → δ=0.05 büyük bir standardize etki (d≈2.5) → **N=45 fazlasıyla yeterli**
- `cooperation_score_avg`: σ≈0.108 → δ=0.05 küçük bir standardize etki (d≈0.46) → **N=45 yetersiz, ~75/koşul gerekirdi**

*Sonuç, rapor için:* Gini-tabanlı karşılaştırmalar güçlü güvenle yapılabilir; cooperation_score-tabanlı karşılaştırmalar etki büyüklüğü + güven aralığıyla, "p<0.05" iddiası olmadan raporlanmalı.

### 18.4. KİLİTLİ BULGU — Cooperation Trait'inde Ters Fidelity

> **Bulgu:** `full_experiment_v1` (45 run) ve doğrulayıcı mikro-A/B testi (20 ek LLM çağrısı, ~$0.04) ile: atanan `cooperation_assigned` trait'i ile gözlenen extraction davranışı arasında **tutarlı ters ilişki** bulundu — yüksek cooperation atanan agent'lar **daha fazla**, düşük cooperation atananlar **daha az** çekiyor. `risk_tolerance_assigned` ise beklenen yönde çalışıyor (yüksek risk → yüksek extraction, tutarlı).

**Doğrulama zinciri (eleme sırasıyla):**
1. İlk gözlem (45 run ortalaması): coop→cooperation_score_avg r=-0.345 (p=0.02) — ama bu metrik `collapse_round` ile confound'lu (r≈0.95) çıktı, round-ortalaması yanıltıcıydı.
2. Round 0'a indirgenince (confound'suz): coop→extraction_fraction r=+0.456 (p=0.0016, n=45); risk→extraction_fraction r=+0.678 (p<0.0001).
3. **Implementasyon hatası ihtimali elendi:** trait atama, agent_id eşlemesi, DB join kontrol edildi — hata yok.
4. **Mock baseline ile karşılaştırma:** mock agent (tasarım niyetini yansıtan formül) coop→score r=+1.0 veriyor — yani sistem "doğru" davranışı biliyor, sadece gerçek LLM farklı davranıyor. **Bu, kodun değil LLM'in davranışının sorumlu olduğunu kanıtlıyor.**
5. **Prompt asimetrisi ihtimali elendi:** `_build_system_prompt` git diff ile karakter karakter kontrol edildi, coop ve risk cümleleri simetrik.
6. **Ham veri kontrolü (declared_max confound'u elendi):** risk=0.8 hücrelerinde `declared_max` her zaman 12.0 (sabit) — yani fraction farkı tamamen `extraction_amount`'tan geliyor, declared_max'ten değil.
7. **Risk×coop etkileşimi ihtimali test edildi ve elendi:** risk=0.2 (düşük, sabit) tutulup sadece coop değiştirilen 10 çağrılık mikro-test: coop=0.2 → ort. extraction=4.56, coop=0.8 → ort. extraction=8.40. Ters etki risk seviyesinden bağımsız, her koşulda var.
8. **Kök neden, justification metinleri okunarak teşhis edildi:** LLM, "cooperation" kavramını "kaynağı az kullanmak" olarak değil, **"kendi payımı sorumlu/sürdürülebilir şekilde tam kullanmak"** olarak yorumluyor. Gerekçelerde "diğer ajanların da faydalanmasını sağlıyorum", "sürdürülebilirliği destekliyorum" gibi ifadelerle yüksek çekimi (declared_max'in ~%70'i) meşrulaştırıyor.

**Akademik çerçeveleme:** Bu bir prompt mühendisliği hatası veya implementasyon bug'ı değil — **kavramsal uyumsuzluk (concept misalignment)**: LLM'in "cooperation" kelimesine atadığı davranışsal çerçeve, deneyin operasyonel tanımıyla (Bölüm 5) örtüşmüyor. `risk_tolerance` için bu uyumsuzluk yok. **Genelleştirilebilir çıkarım: trait fidelity, trait'in türüne göre değişir — doğrudan eylem-yönelimli trait'ler (risk) ile soyut/değer-yüklü trait'ler (cooperation) farklı güvenilirlikte LLM'e aktarılıyor.**

**Bu bulgu kilitlidir, mevcut veri/prompt üzerinde tekrar sorgulanmayacak.** Prompt'u netleştirip (örn. "0=mümkün olduğunca çok çek, 1=mümkün olduğunca az çek, diğerlerinin payını koru" gibi davranışsal terimle) tekrar test etmek istenirse, bu **ayrı bir deney** (`experiment_id` farklı) olarak yapılacak, mevcut `full_experiment_v1` verisi/bulgusu değiştirilmeyecek.

### 18.5. Yapısal Kod Denetimi (Sonuç: Sağlam, Küçük Bir Düzeltme)

18 `.py` dosyası tarandı (salt okuma). Sonuç: **yetim dosya yok**, production zinciri net (`run_full_experiment.py → decision_agent.py → referee_node.py → database.py`), test/debug/analiz dosyaları kategorize edilebilir durumda. Tek gerçek risk: `core/graph.py` ve `run_full_experiment.py`'nin kendi inline graph tanımı arasında **çift tanım** vardı (sessiz tutarsızlık riski) — birleştirildi, tek kaynağa indirildi. Referee'nin 4 sorumluluğu (fizik+şok+metrik+persistence) taşıması not edildi ama mimari kuralına (LLM yok) uygun olduğu için şimdilik dokunulmadı.

### 18.6. Maliyet Defteri (Faz 2 toplamı)

| Aşama | Yaklaşık maliyet |
|---|---|
| Tek-agent test + ilk 5-agent testler | ~$0.05 |
| Ratio kalibrasyonu (validation + revalidation + fine-tune) | ~$0.89 |
| `full_experiment_v1` (45 run) | $6.27 |
| Mikro A/B doğrulama (2 tur, 20 çağrı) | ~$0.04 |
| **Toplam (Haiku 4.5 ile)** | **~$7.25** |

### 18.7. Sırada Ne Var (Bölüm 13'ün uygulanması)

1. **Kontrol grubu agent** (`agents/control_agent.py`) — `mock_agent.py` temel alınarak, LLM kullanmayan sabit-kurallı bir karşılaştırma agent'ı; "LLM davranışı gerçekten kurallı bir bot'tan farklı mı" sorusu.
2. **k-means kümeleme** (`analysis/clustering.py`) — mevcut 45 run'lık veriyle, yeni LLM çağrısı gerektirmez.
3. **İnsan baseline kıyası** (`analysis/human_baseline.py`) — literatür taraması gerektirir, en uzun sürecek parça.
4. Docker ve final Sonnet 4.6 run'ı — yukarıdaki üçü netleşene kadar bilinçli olarak ertelendi.

---

*Bu dosya, projenin tüm sözlü mimari kararlarının yazılı haline getirilmiş halidir. Yeni bir karar verildiğinde bu dosya güncellenmelidir — aksi halde "tek doğruluk kaynağı" ilkesi bozulur.*
