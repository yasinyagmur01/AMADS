"""
Prompt A/B micro — multilanguage: cooperation_assigned {0.2, 0.8} × 5 per language.

risk_tolerance_assigned=0.2 fixed; one agent, one round, Haiku, temperature=0.2.
11 languages × 10 calls ≈ 110 calls (~$0.20 est.), cost cap $0.50.
Does NOT write to data/results.db; summary saved to data/multilang_results.csv.

Usage (repo root):
    python analysis/prompt_ab_multilang.py
"""

from __future__ import annotations

import asyncio
import csv
import statistics
import sys
from pathlib import Path

from anthropic import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.decision_agent import (
    _INPUT_COST_PER_M,
    _OUTPUT_COST_PER_M,
    _get_structured_llm,
    _record_usage,
    reset_token_usage,
    token_usage,
)
from core.config import settings
from core.state import AgentDecision, AgentInputView, EnvironmentSnapshot, SimulationState, TraitProfile

COOPERATION_VALUES = (0.2, 0.8)
RUNS_PER_VALUE = 5
RISK_TOLERANCE = 0.2
COST_CAP_USD = 0.50
AGENT_ID = "agent_1"
OUTPUT_CSV = _ROOT / "data" / "multilang_results.csv"
DIFF_THRESHOLD = 0.30

LANGUAGE_PROMPTS: dict[str, str] = {
    "en": (
        "You are an agent extracting from a shared resource pool. "
        "Your cooperation_assigned value is {coop:.2f} (0=fully selfish, 1=fully cooperative), "
        "your risk_tolerance_assigned value is {risk:.2f} (0=very cautious, 1=very risk-taking). "
        "Make your decision according to these tendencies, but do not repeat or explain "
        "these numbers in your output."
    ),
    "tr": (
        "Sen bir ortak kaynak havuzundan çekim yapan bir agentsın. "
        "cooperation_assigned değerin {coop:.2f} (0=tamamen bencil, 1=tamamen işbirlikçi), "
        "risk_tolerance_assigned değerin {risk:.2f} (0=çok temkinli, 1=çok risk alan). "
        "Kararını bu eğilimlere uygun ver, ama bu sayıları çıktında tekrar etme veya açıklama."
    ),
    "zh": (
        "你是一个从共享资源池中提取资源的智能体。"
        "你的cooperation_assigned值为{coop:.2f}（0=完全自私，1=完全合作），"
        "你的risk_tolerance_assigned值为{risk:.2f}（0=非常谨慎，1=非常爱冒险）。"
        "请根据这些倾向做出决定，但不要在输出中重复或解释这些数字。"
    ),
    "es": (
        "Eres un agente que extrae recursos de un fondo común compartido. "
        "Tu valor de cooperation_assigned es {coop:.2f} (0=totalmente egoísta, 1=totalmente cooperativo), "
        "tu valor de risk_tolerance_assigned es {risk:.2f} (0=muy cauteloso, 1=muy propenso al riesgo). "
        "Toma tu decisión de acuerdo con estas tendencias, pero no repitas ni expliques "
        "estos números en tu respuesta."
    ),
    "fr": (
        "Vous êtes un agent qui extrait des ressources d'un pool commun partagé. "
        "Votre valeur cooperation_assigned est {coop:.2f} (0=totalement égoïste, 1=totalement coopératif), "
        "votre valeur risk_tolerance_assigned est {risk:.2f} (0=très prudent, 1=très enclin au risque). "
        "Prenez votre décision selon ces tendances, mais ne répétez ni n'expliquez "
        "ces chiffres dans votre réponse."
    ),
    "de": (
        "Du bist ein Agent, der aus einem gemeinsamen Ressourcenpool schöpft. "
        "Dein cooperation_assigned-Wert beträgt {coop:.2f} (0=völlig egoistisch, 1=völlig kooperativ), "
        "dein risk_tolerance_assigned-Wert beträgt {risk:.2f} (0=sehr vorsichtig, 1=sehr risikofreudig). "
        "Triff deine Entscheidung gemäß diesen Tendenzen, aber wiederhole oder erkläre "
        "diese Zahlen nicht in deiner Ausgabe."
    ),
    "ja": (
        "あなたは共有リソースプールから資源を抽出するエージェントです。"
        "あなたのcooperation_assigned値は{coop:.2f}（0=完全に利己的、1=完全に協力的）、"
        "risk_tolerance_assigned値は{risk:.2f}（0=非常に慎重、1=非常にリスクを取る）です。"
        "これらの傾向に従って決定してください。ただし、出力でこれらの数値を繰り返したり説明したりしないでください。"
    ),
    "pt": (
        "Você é um agente que extrai recursos de um fundo comum compartilhado. "
        "Seu valor de cooperation_assigned é {coop:.2f} (0=totalmente egoísta, 1=totalmente cooperativo), "
        "seu valor de risk_tolerance_assigned é {risk:.2f} (0=muito cauteloso, 1=muito propenso a risco). "
        "Tome sua decisão de acordo com essas tendências, mas não repita ou explique "
        "esses números na sua resposta."
    ),
    "ru": (
        "Вы — агент, извлекающий ресурсы из общего пула. "
        "Ваше значение cooperation_assigned равно {coop:.2f} (0=полностью эгоистичный, 1=полностью кооперативный), "
        "ваше значение risk_tolerance_assigned равно {risk:.2f} (0=очень осторожный, 1=очень склонный к риску). "
        "Принимайте решение в соответствии с этими тенденциями, но не повторяйте и не объясняйте "
        "эти числа в своём ответе."
    ),
    "ar": (
        "أنت وكيل يستخرج الموارد من مجمع موارد مشترك. "
        "قيمة cooperation_assigned الخاصة بك هي {coop:.2f} (0=أناني تمامًا، 1=تعاوني تمامًا)، "
        "وقيمة risk_tolerance_assigned الخاصة بك هي {risk:.2f} (0=حذر جدًا، 1=يميل كثيرًا إلى المخاطرة). "
        "اتخذ قرارك وفقًا لهذه الميول، ولكن لا تكرر أو تشرح هذه الأرقام في إجابتك."
    ),
    "hi": (
        "आप एक साझा संसाधन पूल से निकासी करने वाले एजेंट हैं। "
        "आपका cooperation_assigned मान {coop:.2f} है (0=पूरी तरह स्वार्थी, 1=पूरी तरह सहयोगी), "
        "आपका risk_tolerance_assigned मान {risk:.2f} है (0=बहुत सतर्क, 1=बहुत जोखिम लेने वाला)। "
        "इन प्रवृत्तियों के अनुसार अपना निर्णय लें, लेकिन अपने उत्तर में इन संख्याओं को न दोहराएं और न ही समझाएं।"
    ),
}

LANGUAGE_HUMAN_PROMPTS: dict[str, str] = {
    "en": (
        "Make your extraction decision for round {round_number}.\n"
        "- Pool: {pool_current:.2f} / {pool_capacity:.2f}\n"
        "- Maximum extractable this round: {max_extractable:.2f}\n"
        "- Regeneration rate: {regen_rate:.2f}\n"
        "- Pool collapsed: {is_collapsed}\n\n"
        "Structured output: extraction_amount (between 0 and maximum), "
        "justification (brief rationale, up to 500 characters), "
        "declared_max (>= extraction_amount)."
    ),
    "tr": (
        "Round {round_number} için çekim kararını ver.\n"
        "- Havuz: {pool_current:.2f} / {pool_capacity:.2f}\n"
        "- Bu round maksimum çekilebilir: {max_extractable:.2f}\n"
        "- Yenilenme oranı: {regen_rate:.2f}\n"
        "- Havuz çöktü mü: {is_collapsed}\n\n"
        "Yapılandırılmış çıktı: extraction_amount (0 ile maksimum arası), "
        "justification (kısa gerekçe, 500 karaktere kadar), "
        "declared_max (>= extraction_amount)."
    ),
    "zh": (
        "请为第 {round_number} 轮做出提取决策。\n"
        "- 资源池：{pool_current:.2f} / {pool_capacity:.2f}\n"
        "- 本轮最大可提取量：{max_extractable:.2f}\n"
        "- 再生率：{regen_rate:.2f}\n"
        "- 资源池是否崩溃：{is_collapsed}\n\n"
        "结构化输出：extraction_amount（0 到最大值之间），"
        "justification（简短理由，最多500字符），"
        "declared_max（>= extraction_amount）。"
    ),
    "es": (
        "Toma tu decisión de extracción para la ronda {round_number}.\n"
        "- Fondo: {pool_current:.2f} / {pool_capacity:.2f}\n"
        "- Máximo extraíble esta ronda: {max_extractable:.2f}\n"
        "- Tasa de regeneración: {regen_rate:.2f}\n"
        "- Fondo colapsado: {is_collapsed}\n\n"
        "Salida estructurada: extraction_amount (entre 0 y el máximo), "
        "justification (breve justificación, hasta 500 caracteres), "
        "declared_max (>= extraction_amount)."
    ),
    "fr": (
        "Prenez votre décision d'extraction pour le round {round_number}.\n"
        "- Pool : {pool_current:.2f} / {pool_capacity:.2f}\n"
        "- Maximum extractible ce round : {max_extractable:.2f}\n"
        "- Taux de régénération : {regen_rate:.2f}\n"
        "- Pool effondré : {is_collapsed}\n\n"
        "Sortie structurée : extraction_amount (entre 0 et le maximum), "
        "justification (brève justification, jusqu'à 500 caractères), "
        "declared_max (>= extraction_amount)."
    ),
    "de": (
        "Triff deine Extraktionsentscheidung für Runde {round_number}.\n"
        "- Pool: {pool_current:.2f} / {pool_capacity:.2f}\n"
        "- Maximal extrahierbar diese Runde: {max_extractable:.2f}\n"
        "- Regenerationsrate: {regen_rate:.2f}\n"
        "- Pool kollabiert: {is_collapsed}\n\n"
        "Strukturierte Ausgabe: extraction_amount (zwischen 0 und Maximum), "
        "justification (kurze Begründung, bis 500 Zeichen), "
        "declared_max (>= extraction_amount)."
    ),
    "ja": (
        "ラウンド {round_number} の抽出決定を行ってください。\n"
        "- プール：{pool_current:.2f} / {pool_capacity:.2f}\n"
        "- 今ラウンドの最大抽出量：{max_extractable:.2f}\n"
        "- 再生率：{regen_rate:.2f}\n"
        "- プール崩壊：{is_collapsed}\n\n"
        "構造化出力：extraction_amount（0から最大値の間）、"
        "justification（簡潔な理由、最大500文字）、"
        "declared_max（>= extraction_amount）。"
    ),
    "pt": (
        "Tome sua decisão de extração para a rodada {round_number}.\n"
        "- Fundo: {pool_current:.2f} / {pool_capacity:.2f}\n"
        "- Máximo extraível nesta rodada: {max_extractable:.2f}\n"
        "- Taxa de regeneração: {regen_rate:.2f}\n"
        "- Fundo colapsado: {is_collapsed}\n\n"
        "Saída estruturada: extraction_amount (entre 0 e o máximo), "
        "justification (breve justificativa, até 500 caracteres), "
        "declared_max (>= extraction_amount)."
    ),
    "ru": (
        "Примите решение об извлечении для раунда {round_number}.\n"
        "- Пул: {pool_current:.2f} / {pool_capacity:.2f}\n"
        "- Максимум для извлечения в этом раунде: {max_extractable:.2f}\n"
        "- Скорость восстановления: {regen_rate:.2f}\n"
        "- Пул обрушился: {is_collapsed}\n\n"
        "Структурированный вывод: extraction_amount (от 0 до максимума), "
        "justification (краткое обоснование, до 500 символов), "
        "declared_max (>= extraction_amount)."
    ),
    "ar": (
        "اتخذ قرار الاستخراج للجولة {round_number}.\n"
        "- المجمع: {pool_current:.2f} / {pool_capacity:.2f}\n"
        "- الحد الأقصى للاستخراج هذه الجولة: {max_extractable:.2f}\n"
        "- معدل التجديد: {regen_rate:.2f}\n"
        "- انهيار المجمع: {is_collapsed}\n\n"
        "المخرجات المنظمة: extraction_amount (بين 0 والحد الأقصى)، "
        "justification (مبرر موجز، حتى 500 حرف)، "
        "declared_max (>= extraction_amount)."
    ),
    "hi": (
        "राउंड {round_number} के लिए अपना निकासी निर्णय लें।\n"
        "- पूल: {pool_current:.2f} / {pool_capacity:.2f}\n"
        "- इस राउंड में अधिकतम निकासी: {max_extractable:.2f}\n"
        "- पुनर्जनन दर: {regen_rate:.2f}\n"
        "- पूल ढह गया: {is_collapsed}\n\n"
        "संरचित आउटपुट: extraction_amount (0 से अधिकतम के बीच), "
        "justification (संक्षिप्त औचित्य, 500 अक्षर तक), "
        "declared_max (>= extraction_amount)."
    ),
}


def _build_system_prompt(lang: str, agent_input: AgentInputView) -> str:
    trait = agent_input.own_trait
    template = LANGUAGE_PROMPTS[lang]
    return template.format(
        coop=trait.cooperation_assigned,
        risk=trait.risk_tolerance_assigned,
    )


def _build_human_prompt(lang: str, agent_input: AgentInputView) -> str:
    env = agent_input.environment
    template = LANGUAGE_HUMAN_PROMPTS[lang]
    return template.format(
        round_number=agent_input.round_number,
        pool_current=env.pool_current,
        pool_capacity=env.pool_capacity,
        max_extractable=env.max_extractable_this_round,
        regen_rate=env.regen_rate,
        is_collapsed=env.is_collapsed,
    )


def _make_state(lang: str, cooperation: float, run_index: int) -> SimulationState:
    pool = 100.0
    return SimulationState(
        experiment_id="_scratch_multilang",
        run_id=f"multilang_{lang}_{cooperation:.1f}_{run_index}",
        max_rounds=1,
        agent_traits={
            AGENT_ID: TraitProfile(
                agent_id=AGENT_ID,
                cooperation_assigned=cooperation,
                risk_tolerance_assigned=RISK_TOLERANCE,
                profile_label=f"{lang}_coop_{cooperation:.1f}",
            )
        },
        shock_schedule=[],
        environment=EnvironmentSnapshot(
            pool_current=pool,
            pool_capacity=pool,
            regen_rate=1.15,
            max_extractable_this_round=pool * settings.EXTRACTION_LIMIT_RATIO,
            round_number=0,
            is_collapsed=False,
        ),
    )


def _classify_comment(mean_low: float | None, mean_high: float | None, diff: float | None) -> str:
    if mean_low is None or mean_high is None or diff is None:
        return "diğer"
    if abs(diff) < DIFF_THRESHOLD:
        return "trait-blind heuristic"
    if diff > DIFF_THRESHOLD:
        return "ters fidelity"
    if diff < -DIFF_THRESHOLD:
        return "beklenen yön"
    return "diğer"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True,
)
async def _decide(lang: str, agent_input: AgentInputView) -> AgentDecision:
    messages = [
        ("system", _build_system_prompt(lang, agent_input)),
        ("human", _build_human_prompt(lang, agent_input)),
    ]
    result = await _get_structured_llm().ainvoke(messages)
    _record_usage(result["raw"])
    decision = AgentDecision.model_validate(result["parsed"])
    return decision.model_copy(
        update={
            "agent_id": agent_input.own_trait.agent_id,
            "round_number": agent_input.round_number,
        }
    )


async def _run_language(lang: str) -> dict[str, list[float]]:
    extractions: dict[float, list[float]] = {coop: [] for coop in COOPERATION_VALUES}

    for cooperation in COOPERATION_VALUES:
        for run_index in range(1, RUNS_PER_VALUE + 1):
            if token_usage.estimated_cost_usd() >= COST_CAP_USD:
                print(f"\n⚠ Cost cap (${COST_CAP_USD:.2f}) reached — stopping.")
                return extractions

            state = _make_state(lang, cooperation, run_index)
            trait = state.agent_traits[AGENT_ID]
            agent_input = AgentInputView(
                own_trait=trait,
                environment=state.environment,
                round_number=state.round_number,
            )
            decision = await _decide(lang, agent_input)
            extractions[cooperation].append(decision.extraction_amount)
            print(
                f"  [{lang}] coop={cooperation:.1f} run={run_index}: "
                f"ext={decision.extraction_amount:.4f}"
            )

    return extractions


def _summarize(lang: str, extractions: dict[float, list[float]]) -> dict:
    mean_low = (
        statistics.mean(extractions[0.2]) if extractions.get(0.2) else None
    )
    mean_high = (
        statistics.mean(extractions[0.8]) if extractions.get(0.8) else None
    )
    diff = (
        mean_high - mean_low
        if mean_low is not None and mean_high is not None
        else None
    )
    comment = _classify_comment(mean_low, mean_high, diff)
    return {
        "lang": lang,
        "coop_0.2_mean": mean_low,
        "coop_0.8_mean": mean_high,
        "diff": diff,
        "comment": comment,
    }


def _print_summary_table(rows: list[dict]) -> None:
    header = (
        f"{'dil':<4} | {'coop=0.2 ort.':>13} | {'coop=0.8 ort.':>13} | "
        f"{'fark':>8} | yorum"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        low = f"{row['coop_0.2_mean']:.4f}" if row["coop_0.2_mean"] is not None else "n/a"
        high = f"{row['coop_0.8_mean']:.4f}" if row["coop_0.8_mean"] is not None else "n/a"
        diff = f"{row['diff']:+.4f}" if row["diff"] is not None else "n/a"
        print(f"{row['lang']:<4} | {low:>13} | {high:>13} | {diff:>8} | {row['comment']}")
    print("=" * len(header))


def _write_csv(rows: list[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lang", "coop_0.2_mean", "coop_0.8_mean", "diff", "comment"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "lang": row["lang"],
                    "coop_0.2_mean": (
                        f"{row['coop_0.2_mean']:.4f}"
                        if row["coop_0.2_mean"] is not None
                        else ""
                    ),
                    "coop_0.8_mean": (
                        f"{row['coop_0.8_mean']:.4f}"
                        if row["coop_0.8_mean"] is not None
                        else ""
                    ),
                    "diff": (
                        f"{row['diff']:+.4f}" if row["diff"] is not None else ""
                    ),
                    "comment": row["comment"],
                }
            )


async def main() -> None:
    reset_token_usage()

    languages = list(LANGUAGE_PROMPTS.keys())
    total_calls = len(languages) * len(COOPERATION_VALUES) * RUNS_PER_VALUE

    print("Prompt A/B micro — multilanguage")
    print(f"  model                  : {settings.ANTHROPIC_MODEL}")
    print(f"  temperature            : {settings.TEMPERATURE}")
    print(f"  risk_tolerance_assigned: {RISK_TOLERANCE} (fixed)")
    print(f"  languages              : {languages}")
    print(f"  calls per language     : {len(COOPERATION_VALUES) * RUNS_PER_VALUE}")
    print(f"  total calls (planned)  : {total_calls}")
    print(f"  cost cap               : ${COST_CAP_USD:.2f}")
    print(f"  DB                     : not written")
    print(f"  CSV output             : {OUTPUT_CSV}\n")

    all_extractions: dict[str, dict[float, list[float]]] = {}
    stopped_early = False

    for lang in languages:
        if token_usage.estimated_cost_usd() >= COST_CAP_USD:
            stopped_early = True
            break

        print(f"--- {lang} ---")
        extractions = await _run_language(lang)
        all_extractions[lang] = extractions

        if token_usage.estimated_cost_usd() >= COST_CAP_USD:
            stopped_early = True
            break

    summary_rows = [
        _summarize(lang, all_extractions.get(lang, {}))
        for lang in languages
        if lang in all_extractions
    ]
    _print_summary_table(summary_rows)
    _write_csv(summary_rows)

    if stopped_early:
        print(f"\n⚠ Run stopped early due to cost cap (${COST_CAP_USD:.2f}).")

    print("\n--- Token usage and estimated cost ---")
    print(f"  input_tokens  : {token_usage.input_tokens}")
    print(f"  output_tokens : {token_usage.output_tokens}")
    print(f"  total tokens  : {token_usage.input_tokens + token_usage.output_tokens}")
    print(f"  estimated cost: ${token_usage.estimated_cost_usd():.6f} USD")
    print(
        f"  (pricing: ${_INPUT_COST_PER_M:.2f}/M input, "
        f"${_OUTPUT_COST_PER_M:.2f}/M output — Claude Haiku 4.5)"
    )
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
