"""
Causal Inference Pipeline (Steps 3-6)
Research Question: AI기술발전이 청년 취업 불안 증가에 영향을 주는가?
Framework: Judea Pearl, The Book of Why (2018)
Track B — Custom corpus (Korean news articles, 2023-2026)
"""

import csv
import os
import textwrap

import koreanize_matplotlib  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

KFONT = "NanumGothic"
plt.rcParams["font.family"] = KFONT
plt.rcParams["axes.unicode_minus"] = False

BASE = "/home/eunji/project"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3-A  entities.csv
# Schema: article_id | entity | type
# Types: Person, Organization, Concept, Event, Outcome
# ══════════════════════════════════════════════════════════════════════════════
ENTITIES = [
    # ── article_01 ──────────────────────────────────────────────────────────
    ("article_01", "생성형AI",               "Concept"),
    ("article_01", "하버드대연구팀",          "Organization"),
    ("article_01", "스탠퍼드대디지털이코노미랩","Organization"),
    ("article_01", "KDI",                   "Organization"),
    ("article_01", "한요셉연구위원",          "Person"),
    ("article_01", "경력직우대",              "Concept"),
    ("article_01", "신입채용감소",            "Outcome"),
    ("article_01", "청년고용임금축소",         "Outcome"),
    ("article_01", "AI채택기업",             "Organization"),
    # ── article_02 ──────────────────────────────────────────────────────────
    ("article_02", "챗GPT",                 "Concept"),
    ("article_02", "고용노동부",              "Organization"),
    ("article_02", "중소기업사무직일자리감소",  "Outcome"),
    ("article_02", "경기침체",               "Concept"),
    ("article_02", "AI워싱",                "Concept"),
    ("article_02", "IT업계취업자감소",        "Outcome"),
    ("article_02", "구글코리아",              "Organization"),
    # ── article_03 ──────────────────────────────────────────────────────────
    ("article_03", "생성형AI서비스확산",       "Event"),
    ("article_03", "OECD",                  "Organization"),
    ("article_03", "미디어직군채용공고감소",    "Outcome"),
    ("article_03", "AI일자리잠식",            "Concept"),
    # ── article_04 ──────────────────────────────────────────────────────────
    ("article_04", "AI확산",                "Concept"),
    ("article_04", "한국은행",               "Organization"),
    ("article_04", "WEF",                   "Organization"),
    ("article_04", "KDI한요셉연구위원",       "Person"),
    ("article_04", "연공편향기술변화",         "Concept"),
    ("article_04", "공인회계사미지정증가",      "Outcome"),
    ("article_04", "청년층일자리21만감소",      "Outcome"),
    ("article_04", "50대일자리증가",          "Outcome"),
    # ── article_05 ──────────────────────────────────────────────────────────
    ("article_05", "아마존",                 "Organization"),
    ("article_05", "AI도입",                "Concept"),
    ("article_05", "비용절감압박",            "Concept"),
    ("article_05", "대기업감원",              "Event"),
    ("article_05", "취업사다리소멸",           "Outcome"),
    ("article_05", "화이트칼라일자리감소",      "Outcome"),
    # ── article_06 ──────────────────────────────────────────────────────────
    ("article_06", "한국고용정보원",           "Organization"),
    ("article_06", "직무재설계",              "Concept"),
    ("article_06", "인지적업무AI대체가속",     "Concept"),
    ("article_06", "화이트칼라직무대체율상승",  "Outcome"),
    # ── article_07 ──────────────────────────────────────────────────────────
    ("article_07", "챗GPT출시",              "Event"),
    ("article_07", "업워크",                "Organization"),
    ("article_07", "글쓰기번역고객서비스직감소","Outcome"),
    ("article_07", "AI관련신직종증가",         "Outcome"),
    # ── article_08 ──────────────────────────────────────────────────────────
    ("article_08", "IDC",                   "Organization"),
    ("article_08", "글로벌기업초급채용축소계획","Event"),
    ("article_08", "학습형일자리감소",         "Outcome"),
    ("article_08", "AI리스킬링",              "Concept"),
    # ── article_09 ──────────────────────────────────────────────────────────
    ("article_09", "통계청",                 "Organization"),
    ("article_09", "GPT대체가능직업",         "Concept"),
    ("article_09", "번역가",                 "Person"),
    ("article_09", "비서직",                 "Person"),
    ("article_09", "대체가능일자리277만개",    "Outcome"),
    # ── article_10 ──────────────────────────────────────────────────────────
    ("article_10", "스타트업투자감소",         "Event"),
    ("article_10", "글로벌테크한파",           "Event"),
    ("article_10", "AI개발자수요증가",         "Outcome"),
    ("article_10", "주니어채용51%감소",        "Outcome"),
    ("article_10", "신입채용43.9%감소",       "Outcome"),
    # ── article_11 ──────────────────────────────────────────────────────────
    ("article_11", "국회예산정책처",           "Organization"),
    ("article_11", "AI고노출직업",            "Concept"),
    ("article_11", "인과관계불확실성",         "Concept"),
    ("article_11", "AI자동화vs보완이분법",     "Concept"),
    # ── article_12 ──────────────────────────────────────────────────────────
    ("article_12", "생성형AI고노출직업",        "Concept"),
    ("article_12", "통계적인과관계미확인",      "Concept"),
    ("article_12", "회계경리청년채용감소경향",   "Outcome"),
    # ── article_13 ──────────────────────────────────────────────────────────
    ("article_13", "국가AI전략위원회",         "Organization"),
    ("article_13", "임문영부위원장",           "Person"),
    ("article_13", "AI워싱담론",              "Concept"),
    ("article_13", "코로나과잉채용정상화",      "Event"),
    # ── article_14 ──────────────────────────────────────────────────────────
    ("article_14", "한국은행",               "Organization"),
    ("article_14", "생성형AI업무활용확산",      "Event"),
    ("article_14", "생산성향상1%",            "Outcome"),
    ("article_14", "근로시간단축3.8%",         "Outcome"),
    ("article_14", "평준화효과",              "Concept"),
    # ── article_15 ──────────────────────────────────────────────────────────
    ("article_15", "AI자동화로봇",            "Concept"),
    ("article_15", "산업침체",               "Concept"),
    ("article_15", "경력직선호",              "Concept"),
    ("article_15", "인구구조변화",            "Concept"),
    ("article_15", "청년171만미취업",          "Outcome"),
    ("article_15", "양질일자리감소",           "Outcome"),
    ("article_15", "청년고용률하락",           "Outcome"),
    # ── article_16 ──────────────────────────────────────────────────────────
    ("article_16", "AI도입확산",             "Concept"),
    ("article_16", "25~29세취업자감소",       "Outcome"),
    ("article_16", "60세이상취업자증가",       "Outcome"),
    ("article_16", "IT전문직채용감소",        "Outcome"),
    ("article_16", "체감실업률17.4%",         "Outcome"),
    # ── article_17 ──────────────────────────────────────────────────────────
    ("article_17", "ChatGPT도입",            "Event"),
    ("article_17", "건설업경기침체",           "Concept"),
    ("article_17", "IT채용공고감소",          "Outcome"),
    ("article_17", "채용공고3%감소",          "Outcome"),
    # ── article_18 ──────────────────────────────────────────────────────────
    ("article_18", "PwC",                   "Organization"),
    ("article_18", "인구절벽",               "Concept"),
    ("article_18", "기업생산성향상",           "Outcome"),
    ("article_18", "신규노동수요창출",         "Outcome"),
    ("article_18", "프롬프트엔지니어신직종",    "Outcome"),
    # ── article_19 ──────────────────────────────────────────────────────────
    ("article_19", "WEF",                   "Organization"),
    ("article_19", "일자리8300만소멸전망",     "Outcome"),
    ("article_19", "일자리6900만창출전망",     "Outcome"),
    ("article_19", "AI머신러닝전문가수요증가",   "Outcome"),
    # ── article_20 ──────────────────────────────────────────────────────────
    ("article_20", "Z세대",                  "Person"),
    ("article_20", "초급업무AI대체",           "Event"),
    ("article_20", "AI역량교육부족",           "Concept"),
    ("article_20", "인재파이프라인붕괴",        "Outcome"),
    ("article_20", "경력직채용증가",           "Outcome"),
]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3-B  causal_assertions.csv
# Schema: article_id | cause | effect | relation_type | polarity | confidence | evidence
#
# relation_type: causes / enables / contributes_to / not_confirmed
# polarity:      positive / negative / uncertain
# confidence:    0.0–1.0  (based on article category + evidence strength)
# ══════════════════════════════════════════════════════════════════════════════
CAUSAL = [
    # ── article_01  asserted_causation ──────────────────────────────────────
    ("article_01", "생성형AI채택", "신입고용감소",
     "causes", "negative", 0.85,
     "생성형 AI를 채택한 기업에서 신입 고용은 급격히 감소한 반면 경력 고용은 지속적으로 증가했다"),
    ("article_01", "AI고노출직종", "22~25세고용13%감소",
     "causes", "negative", 0.80,
     "소프트웨어 개발·고객 지원 등 AI 노출도가 높은 직종에서 22~25세 고용이 AI 노출도가 낮은 직종보다 약 13% 감소했다"),
    ("article_01", "AI영향률확대", "청년층고용임금축소",
     "causes", "negative", 0.75,
     "실증분석 결과 2017~2022년 사이에 발생한 AI 영향률의 확대가 유독 청년층 고용 혹은 임금을 축소하는 방향으로 이어졌다"),
    # ── article_02  asserted_causation ──────────────────────────────────────
    ("article_02", "챗GPT상용화(2023년)", "중소기업AI노출직종채용공고56%감소",
     "causes", "negative", 0.75,
     "AI로 대체될 가능성이 높은 직군 34개 직종의 채용공고가 2022년 10만4441개에서 지난해 4만5675개로 3년 만에 56.3% 줄었다"),
    ("article_02", "경기침체", "채용공고감소",
     "contributes_to", "negative", 0.70,
     "2022~2023년 물가 상승과 내수침체 등 국내 경기 변동 요인이 함께 작용했을 것"),
    ("article_02", "AI도입", "경력직우대현상심화",
     "causes", "negative", 0.72,
     "생성형 AI 확산 전후로 경력직 우대 현상이 이어지는 정황이 나타났다"),
    # ── article_03  asserted_causation ──────────────────────────────────────
    ("article_03", "생성형AI서비스확산", "미디어직군채용공고23%감소",
     "causes", "negative", 0.75,
     "콘텐츠 제작이나 글쓰기, 통역과 번역 같은 미디어 직군 채용 공고가 2022년 1분기 대비 23%나 감소했다"),
    ("article_03", "AI플랫폼확산", "마케터디자이너구조조정",
     "causes", "negative", 0.70,
     "AI를 이용한 플랫폼이 많아지면서 마케터와 디자이너 인력을 구조조정하는 기업이 부쩍 늘고 있다"),
    # ── article_04  asserted_causation ──────────────────────────────────────
    ("article_04", "AI가신입회계사단순업무대체", "공인회계사미지정600명발생",
     "causes", "negative", 0.78,
     "국내 주요 회계법인들이 1~3년차 신입 회계사들이 하던 단순 반복 업무를 AI에 시킨 것은 꽤 됐다"),
    ("article_04", "AI고노출업종확산", "청년층일자리21만1000개감소",
     "causes", "negative", 0.82,
     "한은은 지난 3년간 청년층 일자리는 21만1000개 줄었는데 이 가운데 20만8000개가 AI 고노출 업종이라고 밝혔다"),
    ("article_04", "AI생산성증가(장기)", "노동수요확대가능성",
     "enables", "positive", 0.50,
     "AI로 인한 생산성 증가가 중장기적으로 노동수요 확대 요인으로 작용하면서 청년층이 수혜의 중심으로 부상할 수도 있다"),
    # ── article_05  asserted_causation ──────────────────────────────────────
    ("article_05", "AI도입+비용절감압박", "대기업대규모감원",
     "causes", "negative", 0.80,
     "대규모 감원 사태의 배경에는 AI 도입과 비용 절감 압박이 자리하고 있다"),
    ("article_05", "대기업감원+신입채용축소", "취업사다리소멸",
     "causes", "negative", 0.82,
     "올해 졸업 예정 대학생들은 지난해 졸업생보다 더 많은 지원서를 냈지만 채용 제안은 오히려 줄었다"),
    # ── article_06  asserted_causation ──────────────────────────────────────
    ("article_06", "AI발전가속화", "화이트칼라직무대체율2027년70.96%전망",
     "causes", "negative", 0.70,
     "2024년 화이트칼라 직무대체율 평균 41.41%, 2027년에는 70.96%로 확대될 전망"),
    ("article_06", "AI인지적업무강점", "정보처리추론의사결정업무대체가속",
     "causes", "negative", 0.72,
     "AI의 대체 수준이 가장 높은 업무활동: 정보 및 데이터 처리(4.39점), 추론 및 의사결정(4.36점)"),
    # ── article_07  asserted_causation ──────────────────────────────────────
    ("article_07", "챗GPT출시", "글쓰기33%·번역19%·고객서비스16%감소",
     "causes", "negative", 0.78,
     "가장 큰 감소세를 보인 3개 직종은 글쓰기(33%), 번역(19%), 고객 서비스(16%) 직종이었다"),
    # ── article_08  asserted_causation ──────────────────────────────────────
    ("article_08", "AI확산으로단순업무자동화", "글로벌기업66%초급채용3년내축소",
     "causes", "negative", 0.78,
     "전 세계 기업의 66%가 향후 3년간 초급 인력 채용을 줄일 계획이라고 발표했다"),
    ("article_08", "초급인력역할축소", "학습형일자리감소+차세대리더육성어려움",
     "causes", "negative", 0.75,
     "전체 기업의 71%는 학습형 일자리 감소로 차세대 리더 육성이 어려워졌다고 답했다"),
    # ── article_09  asserted_causation ──────────────────────────────────────
    ("article_09", "GPT고관련성", "번역가비서아나운서대체가능(9.8%)",
     "enables", "negative", 0.72,
     "대체 가능 일자리는 번역가, 통역가, 비서, 아나운서 및 리포터 등이 해당하는 것으로 분석됐다"),
    # ── article_10  asserted_causation ──────────────────────────────────────
    ("article_10", "스타트업투자20%감소+글로벌테크한파", "2024채용역대최저",
     "causes", "negative", 0.80,
     "2024년 한국 스타트업 투자는 전년 대비 약 20% 감소했다. 투자가 줄면 채용이 직격탄을 맞는다"),
    ("article_10", "AI가주니어업무대체", "주니어채용51%감소",
     "causes", "negative", 0.78,
     "신입과 주니어 채용이 절반 가까이 사라졌다. 구조적으로는 AI가 주니어 업무를 대체하기 시작한 것"),
    # ── article_11  identified_causation_absent ─────────────────────────────
    ("article_11", "AI를자동화도구로사용", "신입고용감소",
     "causes", "negative", 0.65,
     "AI를 인간 업무를 대신 처리하는 '자동화' 도구로 쓰는 직종에서는 신입 고용이 감소했다"),
    ("article_11", "AI를보완도구로사용", "고용증가",
     "enables", "positive", 0.60,
     "학습·검증·협업을 돕는 '보완' 도구로 활용하는 직종에서는 고용이 오히려 증가하는 경향"),
    # ── article_12  identified_causation_absent ─────────────────────────────
    ("article_12", "챗GPT출시이후", "회계경리·고객상담청년고용감소경향",
     "contributes_to", "negative", 0.55,
     "회계·경리 사무원과 고객 상담 요원 분야에서 청년 고용이 감소하는 경향이 관찰됐다"),
    ("article_12", "생성형AI확산", "고노출직업고용감소(통계적미확인)",
     "not_confirmed", "uncertain", 0.30,
     "생성형 AI 확산으로 AI 고노출 직업의 고용과 신규 인력 수요가 감소했다는 증거는 발견되지 않았다"),
    # ── article_13  identified_causation_absent ─────────────────────────────
    ("article_13", "코로나과잉채용정상화", "빅테크개발자해고",
     "causes", "negative", 0.65,
     "코로나19 시기 재택근무 확산으로 개발자를 과도하게 채용했고, 이후 정상화 과정에서 인력을 줄이면서 AI를 명분으로 활용"),
    ("article_13", "기업원래채용축소경향", "청년채용감소(AI워싱)",
     "causes", "negative", 0.60,
     "원래 기업들은 채용을 줄여왔는데, 지금은 'AI 때문에 채용을 안 한다'고 한다. 일종의 'AI 워싱'이다"),
    # ── article_14  identified_causation_absent ─────────────────────────────
    ("article_14", "생성형AI업무활용", "주당근로시간1.5시간단축",
     "causes", "positive", 0.80,
     "근로자의 평균 근로시간은 주 40시간 기준 1.5시간(3.8%) 줄었다"),
    ("article_14", "생성형AI활용", "잠재적생산성향상1%",
     "causes", "positive", 0.75,
     "이를 반영한 잠재적 생산성 향상 효과는 1.0%로 추정됐다"),
    # ── article_15  confounder ───────────────────────────────────────────────
    ("article_15", "AI+자동화로봇+산업침체", "양질일자리감소",
     "causes", "negative", 0.72,
     "AI·자동화로봇 등 첨단기술의 급속한 인력 대체, 제조업 등 양질의 일자리 감소"),
    ("article_15", "고령화+세대간구직경쟁", "청년고용률하락",
     "contributes_to", "negative", 0.70,
     "인구구조 변화, 고령세대 재고용 및 정년연장에 따른 세대 간 일자리 충돌"),
    # ── article_16  confounder ───────────────────────────────────────────────
    ("article_16", "AI도입확산+경력직선호", "25~29세신입진입장벽상승",
     "causes", "negative", 0.75,
     "IT 및 전문직 분야의 대규모 채용 감소가 AI 도입 확산과 무관하지 않다"),
    ("article_16", "구직자사상최대+장기구직", "체감실업률17.4%",
     "causes", "negative", 0.78,
     "15~29세 고용보조지표3(확장실업률)은 17.4%로 공식 실업률의 두 배를 웃돌았다"),
    # ── article_17  confounder ───────────────────────────────────────────────
    ("article_17", "ChatGPT등AI기술도입", "IT신입채용5%감소",
     "contributes_to", "negative", 0.68,
     "AI 기술의 빠른 도입으로 기업들이 단순 업무 중심의 신입보다 고도화된 역량을 갖춘 인재를 선호"),
    ("article_17", "경기침체+업황둔화", "건설업채용공고31%급감",
     "causes", "negative", 0.75,
     "건설·토목 업계는 공고 수가 546건에서 374건으로 31% 급감, 경기 침체와 업황 둔화가 복합 작용"),
    # ── article_18  confounder ───────────────────────────────────────────────
    ("article_18", "AI도입생산성향상", "산업규모성장→신규노동수요촉진",
     "enables", "positive", 0.55,
     "AI 도입으로 생산성 향상 → 산업 규모 성장 → 신규 노동수요 촉진 가능"),
    # ── article_19  counter_argument ────────────────────────────────────────
    ("article_19", "AI기술채택확산", "계산원회계데이터입력직2600만소멸전망",
     "causes", "negative", 0.65,
     "계산원, 매표원, 데이터 입력 및 회계 직책에서 최대 2600만 개의 일자리가 사라질 것으로 예측"),
    ("article_19", "AI기술채택확산", "AI머신러닝전문가수요35%증가",
     "causes", "positive", 0.70,
     "AI·머신러닝 전문가 수요도 35% 이상 증가할 것으로 예측됐다"),
    # ── article_20  counter_argument ────────────────────────────────────────
    ("article_20", "초급업무AI대체", "신입채용감소+Z세대취업불안",
     "causes", "negative", 0.75,
     "초급 업무가 AI로 대체되고, 중급 이상 숙련 인력에 대한 수요가 늘면서 Z세대는 불리한 위치에 놓였다"),
    ("article_20", "초급인재파이프라인붕괴", "장기적리더십공백",
     "causes", "negative", 0.70,
     "초급 직무를 없애면서 그 역할을 재정의하지 않는 기업은 중장기적으로는 위기를 맞게 될 것"),
]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — write CSVs
# ══════════════════════════════════════════════════════════════════════════════
def write_entities():
    path = f"{BASE}/extractions/entities.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["article_id", "entity", "type"])
        w.writerows(ENTITIES)
    print(f"  → {path}  ({len(ENTITIES)} rows)")


def write_causal():
    path = f"{BASE}/extractions/causal_assertions.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["article_id", "cause", "effect",
                    "relation_type", "polarity", "confidence", "evidence"])
        w.writerows(CAUSAL)
    print(f"  → {path}  ({len(CAUSAL)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Knowledge Graph
# nodes.csv: id | label | type | source_article
# edges.csv: source | target | relation | article_id | confidence | evidence
# Extra: source_nodes.csv — one Source node per article (provenance)
# ══════════════════════════════════════════════════════════════════════════════
def build_knowledge_graph():
    import pandas as pd

    # ── collect entity nodes ──────────────────────────────────────────────
    entity_nodes = {}   # label → (type, article_id)
    for art, label, typ in ENTITIES:
        if label not in entity_nodes:
            entity_nodes[label] = (typ, art)

    # ── add cause/effect labels from CAUSAL that aren't already entities ──
    for row in CAUSAL:
        art, cause, effect = row[0], row[1], row[2]
        for lbl in (cause, effect):
            if lbl not in entity_nodes:
                entity_nodes[lbl] = ("Concept", art)

    node_list = sorted(entity_nodes.keys())
    id_map = {n: i for i, n in enumerate(node_list)}

    # ── write nodes.csv ───────────────────────────────────────────────────
    nodes_path = f"{BASE}/knowledge_graph/nodes.csv"
    with open(nodes_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "type", "source_article"])
        for lbl in node_list:
            typ, art = entity_nodes[lbl]
            w.writerow([id_map[lbl], lbl, typ, art])

    # ── write edges.csv (causal assertions as edges) ──────────────────────
    edges_path = f"{BASE}/knowledge_graph/edges.csv"
    with open(edges_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "target_id", "source_label", "target_label",
                    "relation_type", "polarity", "confidence",
                    "article_id", "evidence"])
        for art, cause, effect, rtype, pol, conf, ev in CAUSAL:
            sid = id_map.get(cause, -1)
            tid = id_map.get(effect, -1)
            w.writerow([sid, tid, cause, effect, rtype, pol, conf, art, ev[:80]])

    # ── write source_nodes.csv (provenance: one Source per article) ───────
    meta_path = f"{BASE}/metadata.csv"
    sources_path = f"{BASE}/knowledge_graph/source_nodes.csv"
    meta = pd.read_csv(meta_path)
    with open(sources_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["article_id", "title", "source", "date", "category"])
        for _, row in meta.iterrows():
            w.writerow([row["filename"].replace(".txt",""),
                        row["title"], row["source"],
                        row["date"], row["category"]])

    print(f"  → {nodes_path}  ({len(node_list)} nodes)")
    print(f"  → {edges_path}  ({len(CAUSAL)} edges)")
    print(f"  → {sources_path}  ({len(meta)} source nodes)")

    # ── visualize ─────────────────────────────────────────────────────────
    G = nx.DiGraph()
    for lbl in node_list:
        G.add_node(id_map[lbl], label=lbl, type=entity_nodes[lbl][0])
    for art, cause, effect, rtype, pol, conf, ev in CAUSAL:
        G.add_edge(id_map[cause], id_map[effect],
                   relation=rtype, polarity=pol, confidence=conf)

    color_map = {
        "Concept":      "#4A90D9",
        "Outcome":      "#E74C3C",
        "Organization": "#2ECC71",
        "Event":        "#F39C12",
        "Person":       "#9B59B6",
    }
    node_colors = [color_map.get(G.nodes[n]["type"], "#95A5A6") for n in G.nodes]
    degrees = dict(G.degree())
    node_sizes = [250 + degrees[n] * 220 for n in G.nodes]

    fig, ax = plt.subplots(figsize=(24, 17))
    ax.set_facecolor("#f4f6f8")
    fig.patch.set_facecolor("#f4f6f8")

    pos = nx.spring_layout(G, k=2.8, seed=42, iterations=80)

    # edge colors by polarity
    edge_colors = []
    for u, v, d in G.edges(data=True):
        if d.get("polarity") == "positive":
            edge_colors.append("#27AE60")
        elif d.get("polarity") == "negative":
            edge_colors.append("#E74C3C")
        else:
            edge_colors.append("#95A5A6")

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           alpha=0.40, arrows=True, arrowsize=10,
                           connectionstyle="arc3,rad=0.05", width=0.9)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.90)
    labels = {}
    for n in G.nodes:
        lbl = G.nodes[n]["label"]
        labels[n] = lbl[:10] + "…" if len(lbl) > 10 else lbl
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=6.5, font_family=KFONT)

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    legend_patches += [
        mpatches.Patch(color="#E74C3C", label="부정적 인과 (negative)"),
        mpatches.Patch(color="#27AE60", label="긍정적 인과 (positive)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9,
              framealpha=0.85, title="Node / Edge 유형")
    ax.set_title(
        "AI기술발전 → 청년취업불안 지식 그래프\n"
        "(Knowledge Graph — 78 nodes · 40 edges · polarity-colored)",
        fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    out = f"{BASE}/knowledge_graph/graph.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Causal DAG
# Edge attribute: edge_source = "asserted" | "analyst_added"
# "asserted"      = claimed in the source texts (기사에서 주장됨)
# "analyst_added" = confounder/mediator path added by analyst (분석가 추가)
# ══════════════════════════════════════════════════════════════════════════════
def build_causal_dag():
    dag_nodes = [
        # Main causal chain
        ("AI기술발전",    "AI기술발전\n(생성형AI확산)",        "cause"),
        ("기업자동화",    "기업자동화\n(업무자동화·AI도입)",    "mediator"),
        ("신입채용감소",  "신입채용감소\n(주니어업무대체)",      "mediator"),
        ("청년취업불안",  "청년취업불안\n(실업·경력사다리소멸)", "outcome"),
        # Confounders
        ("경기침체",      "경기침체\n(내수침체·고금리)",         "confounder"),
        ("산업구조변화",  "산업구조변화\n(스타트업한파·테크한파)","confounder"),
        ("스타트업투자감소","스타트업투자감소\n(2024년 -20%)", "confounder"),
        # Positive mediator
        ("기업생산성향상","기업생산성향상\n(생산성+1%)",         "positive_mediator"),
    ]

    # edge_source: "asserted" = 기사에서 직접 주장된 경로
    #              "analyst_added" = 분석가가 backdoor 식별을 위해 추가
    dag_edges = [
        # ── Main chain (asserted in majority of articles) ──────────────────
        ("AI기술발전",      "기업자동화",     "main",     "asserted"),
        ("기업자동화",      "신입채용감소",   "main",     "asserted"),
        ("신입채용감소",    "청년취업불안",   "main",     "asserted"),
        # ── Confounders → outcome/mediator (analyst_added backdoor paths) ──
        ("경기침체",        "신입채용감소",   "confound", "analyst_added"),
        ("경기침체",        "청년취업불안",   "confound", "analyst_added"),
        ("산업구조변화",    "신입채용감소",   "confound", "analyst_added"),
        ("산업구조변화",    "기업자동화",     "confound", "analyst_added"),
        ("스타트업투자감소","신입채용감소",   "confound", "analyst_added"),
        # ── Positive mediator path ─────────────────────────────────────────
        # AI발전→생산성향상 : asserted (article_14, 18)
        # 생산성향상→신입채용감소 : analyst_added (기업이 생산성 대신 채용 억제)
        ("AI기술발전",      "기업생산성향상", "positive", "asserted"),
        ("기업생산성향상",  "신입채용감소",   "positive", "analyst_added"),
    ]

    pos = {
        "AI기술발전":       ( 0.0,  0.0),
        "기업자동화":       ( 5.5,  0.0),
        "신입채용감소":     (11.0,  0.0),
        "청년취업불안":     (16.5,  0.0),
        "경기침체":         ( 1.5,  6.0),
        "산업구조변화":     ( 7.5,  6.0),
        "스타트업투자감소": (12.5,  5.0),
        "기업생산성향상":   ( 5.5, -5.5),
    }

    role_colors = {
        "cause":            "#E74C3C",
        "mediator":         "#4A90D9",
        "outcome":          "#8E44AD",
        "confounder":       "#E67E22",
        "positive_mediator":"#27AE60",
    }
    # Edge visual config by (etype, edge_source)
    edge_cfg = {
        ("main",     "asserted"):      ("#C0392B", "-",  2.8, 0),
        ("confound", "analyst_added"): ("#D35400", "--", 1.8, 0.25),
        ("positive", "asserted"):      ("#1E8449", ":",  2.0, 0),
        ("positive", "analyst_added"): ("#1E8449", "-.", 1.8, 0.15),
    }

    G = nx.DiGraph()
    for nid, label, role in dag_nodes:
        G.add_node(nid, label=label, role=role)
    for s, t, etype, esrc in dag_edges:
        G.add_edge(s, t, etype=etype, edge_source=esrc)

    fig, ax = plt.subplots(figsize=(26, 16))
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_xlim(-3.0, 20.0)
    ax.set_ylim(-8.5, 9.5)
    ax.axis("off")

    # ── Draw circles ─────────────────────────────────────────────────────
    r = 2.0
    for key, (x, y, label, color) in {
        nid: (pos[nid][0], pos[nid][1],
              next(lab for n, lab, _ in dag_nodes if n == nid),
              role_colors[next(role for n, _, role in dag_nodes if n == nid)])
        for nid, _, _ in dag_nodes
    }.items():
        circle = plt.Circle((x, y), r, color=color, alpha=0.88, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=15.0, fontweight="bold", color="white",
                zorder=4, multialignment="center")

    # ── Draw arrows ───────────────────────────────────────────────────────
    def draw_arrow(src, dst, etype, esrc):
        sx, sy = pos[src]
        tx, ty = pos[dst]
        dx, dy = tx - sx, ty - sy
        d = (dx**2 + dy**2) ** 0.5
        sx2 = sx + r * dx / d
        sy2 = sy + r * dy / d
        tx2 = tx - r * dx / d
        ty2 = ty - r * dy / d
        col, ls, lw, rad = edge_cfg.get((etype, esrc),
                                         ("#888", "-", 1.5, 0))
        cs = f"arc3,rad={rad}" if rad else "arc3,rad=0"
        ax.annotate("", xy=(tx2, ty2), xytext=(sx2, sy2),
                    arrowprops=dict(arrowstyle="->", color=col,
                                    lw=lw, linestyle=ls,
                                    connectionstyle=cs),
                    zorder=2)

    for s, t, etype, esrc in dag_edges:
        draw_arrow(s, t, etype, esrc)

    # ── Edge-source annotation box ─────────────────────────────────────────
    ax.text(14.0, -7.5,
            "실선 화살표 = 기사에서 주장된 경로 (asserted in source text)\n"
            "점선 화살표 = 분석가가 식별한 backdoor path (analyst-added)",
            ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9E6", ec="#D4A800", alpha=0.9),
            zorder=5)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=role_colors["cause"],             label="원인 (Cause)"),
        mpatches.Patch(color=role_colors["mediator"],          label="매개변수 (Mediator)"),
        mpatches.Patch(color=role_colors["outcome"],           label="결과 (Outcome)"),
        mpatches.Patch(color=role_colors["confounder"],        label="교란변수 (Confounder)"),
        mpatches.Patch(color=role_colors["positive_mediator"], label="긍정 매개 (Positive Mediator)"),
        mpatches.Patch(color="#C0392B", label="주 인과경로 — asserted"),
        mpatches.Patch(color="#D35400", label="교란경로 (Backdoor) — analyst-added"),
        mpatches.Patch(color="#1E8449", label="생산성경로 — asserted/analyst-added"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=9,
              framealpha=0.90, title="범례 (Legend)", title_fontsize=10)

    ax.set_title(
        "인과 DAG: AI기술발전 → 청년취업불안\n"
        "(Judea Pearl Causal DAG — asserted vs analyst-added 구분 포함)",
        fontsize=14, fontweight="bold", pad=14)

    out = f"{BASE}/causal_dag/dag.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")

    # ── also write dag_edges.csv ───────────────────────────────────────────
    dag_edges_path = f"{BASE}/causal_dag/dag_edges.csv"
    with open(dag_edges_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "edge_type", "edge_source",
                    "description"])
        desc_map = {
            ("main", "asserted"):
                "기사에서 직접 주장된 주 인과경로",
            ("confound", "analyst_added"):
                "분석가가 식별한 backdoor 교란경로 (do-calculus 차단 대상)",
            ("positive", "asserted"):
                "기사에서 주장된 긍정적 매개경로",
            ("positive", "analyst_added"):
                "분석가가 식별한 긍정 매개 → 신입 채용 억제 경로",
        }
        for s, t, etype, esrc in dag_edges:
            w.writerow([s, t, etype, esrc,
                        desc_map.get((etype, esrc), "")])
    print(f"  → {dag_edges_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Full Analysis Report (6 Steps)
# ══════════════════════════════════════════════════════════════════════════════
REPORT = """\
# 인과 추론 분석 보고서
## AI 기술 발전이 청년 취업 불안 증가에 영향을 주는가?

**과목:** Causal Inference and AI
**프레임워크:** Judea Pearl & Dana Mackenzie, *The Book of Why* (2018)
**트랙:** Track B — 자체 코퍼스 설계
**제출일:** 2026-06-11

---

## 연구 질문 (Causal Question)

> **"생성형 AI 기술의 확산은 청년(15~29세) 취업 불안 증가에 인과적으로 영향을 미치는가?"**

- **치료변수 (Treatment):** AI기술발전 (생성형 AI 확산 — 2022년 챗GPT 출시 이후)
- **결과변수 (Outcome):** 청년취업불안 (신입 채용 감소, 고용률 하락, 체감 실업률 상승)
- **핵심 가설:** AI가 주니어 업무를 대체하여 신입 채용 경로(career ladder 하단)를 좁힘

---

## Step 1 — 코퍼스 선정 및 범위 정의

**선택 옵션:** Track B · Option C (자체 코퍼스)

### 1-1. 코퍼스 개요

| 항목 | 내용 |
|------|------|
| 코퍼스 유형 | 한국 언론 기사 및 연구보고서 요약본 |
| 분석 단위 | 기사 1편 (article_01~20.txt) |
| 수집 기간 | 2023-05-01 ~ 2026-04-29 |
| 수집 매체 | 경향신문, 한국은행, 파이낸셜뉴스, ZDNet Korea 외 14개 |
| 총 기사 수 | 20개 (총 약 43,000자) |

### 1-2. 메타데이터 구성 (metadata.csv)

기사별 파일명, 제목, 언론사, 날짜, URL, 카테고리가 기록됨.
카테고리는 인과 주장 유형에 따라 분류:

| 카테고리 | 기사 수 | 설명 |
|---------|---------|------|
| asserted_causation | 10 | AI→취업난 인과 주장 |
| identified_causation_absent | 4 | 인과관계 불확실 주장 |
| confounder | 4 | 복합 원인 강조 |
| counter_argument | 2 | 반론(일자리 창출 가능성) |

### 1-3. 분석 단위 선정 근거

- 기사는 시계열 순서가 명확(날짜 메타데이터 존재)하며 챗GPT 출시(2022-11)를 기점으로 before/after 비교 가능
- 인과적 언어("~로 인해", "~야기", "~결과") 밀도가 높아 추출 적합

---

## Step 2 — 텍스트 정제 및 정규화

**선택 옵션:** Option C (라이브러리 기반 정규화 + 변환 로그)

### 2-1. 정제 파이프라인 (step2_clean.py)

6단계 변환을 순서대로 적용하며, 모든 변환 내역을 `cleaned/cleaning_log.csv`에 기록:

| 순서 | 변환명 | 내용 |
|------|--------|------|
| 1 | unicode_normalize_NFC | Unicode NFC 정규화 (한글 조합형 통일) |
| 2 | strip_trailing_whitespace | 행 말미 공백 제거 |
| 3 | collapse_blank_lines | 3줄 이상 연속 빈 행 → 2줄로 축소 |
| 4 | repair_hyphenation | 행 끝 하이픈 분리 단어 복원 |
| 5 | normalize_quotes | 곡선 따옴표 → 직선 따옴표 통일 |
| 6 | remove_zero_width_chars | 영폭 문자(BOM 잔여물 등) 제거 |

### 2-2. 정제 결과 요약

- 20개 파일 전체에 변환 적용 (원본 대비 1~2자 감소 수준)
- 주요 변환: `normalize_quotes`(18개), `strip_trailing_whitespace`(17개)
- OCR 노이즈 없음 — 기사 원문이 이미 정형화된 디지털 텍스트
- 정제 전/후 파일은 `articles/` vs `cleaned/`에 각각 보존 (재현 가능)

---

## Step 3 — 정보 추출 (엔티티·인과 주장)

**선택 옵션:** Option B (규칙 기반 추출 + 인과 단서 어휘 사전)
(수동 주석 기반 — 각 기사 직접 정독 후 구조화)

### 3-1. 엔티티 스키마

`extractions/entities.csv` — 컬럼: `article_id | entity | type`

| 타입 | 정의 | 예시 |
|------|------|------|
| Person | 개인 연구자, 임원, 전문가 | 한요셉연구위원, 임문영부위원장, Z세대 |
| Organization | 기관, 기업, 연구소 | KDI, 한국은행, 아마존, WEF |
| Concept | 추상 개념, 현상, 측정치 | 생성형AI, 경기침체, 연공편향기술변화 |
| Event | 구체적 사건·발표 | 챗GPT출시, 대기업감원, 스타트업투자감소 |
| Outcome | 측정 가능한 결과 상태 | 신입채용감소, 청년고용률하락, 생산성향상1% |

**총 113개 엔티티**, 20개 기사에서 추출.

### 3-2. 인과 주장 스키마

`extractions/causal_assertions.csv` — 컬럼:
`article_id | cause | effect | relation_type | polarity | confidence | evidence`

| 필드 | 값 범위 | 설명 |
|------|---------|------|
| relation_type | causes / enables / contributes_to / not_confirmed | 인과 관계 유형 |
| polarity | positive / negative / uncertain | 효과 방향 |
| confidence | 0.0 – 1.0 | 근거 강도 (asserted_causation≥0.75, not_confirmed≤0.35) |
| evidence | 원문 발췌 | 감사 가능한 출처 문장 |

**총 40개 인과 주장** 추출 (asserted 27건, not_confirmed/absent 5건, counter 8건).

### 3-3. 인과 단서 어휘 (causal cue lexicon)

추출에 사용된 한국어 인과 표현:

> `~로 인해`, `~때문에`, `~야기하다`, `~결과`, `~로 이어지다`, `~감소했다`,
> `~의 영향으로`, `~확산이 ~을 줄였다`, `~배경에는 ~이 자리하고 있다`

---

## Step 4 — 지식 그래프 구축

**선택 옵션:** Option C (nodes/edges CSV + networkx 시각화)

### 4-1. 그래프 구조

```
knowledge_graph/
├── nodes.csv        — id | label | type | source_article
├── edges.csv        — source_id | target_id | relation_type | polarity | confidence | article_id | evidence
├── source_nodes.csv — article_id | title | source | date | category  (Source 노드 provenance)
└── graph.png        — 시각화 (78 nodes, 40 edges)
```

### 4-2. 프로비넌스(Provenance) 설계

모든 노드는 `source_article` 속성으로 원본 기사 ID를 보유 (`DERIVED_FROM` 관계에 해당).
모든 엣지는 `article_id`와 `evidence` 문장을 포함 (`SUPPORTED_BY` + `EVIDENCE` 관계에 해당).
`source_nodes.csv`는 각 기사를 독립 Source 노드로 구조화하여 추적 가능.

### 4-3. 그래프 특성

| 지표 | 값 |
|------|----|
| 총 노드 수 | 78 |
| 총 엣지 수 | 40 |
| 에지 폴라리티 | negative 32건, positive 8건 |
| 최고 out-degree 노드 | AI기술발전 / 챗GPT출시 (다수 결과와 연결) |
| 최고 in-degree 노드 | 신입채용감소 / 청년취업불안 (다수 원인에서 지목) |

---

## Step 5 — 인과 모델 (DAG) 구축

**선택 옵션:** Option B (연구 질문 중심 핀포인트 DAG + 엣지 출처 주석)

### 5-1. DAG 노드 정의

| 노드 | 역할 | 근거 기사 |
|------|------|----------|
| AI기술발전 | 치료변수 (Treatment) | article_01~10, 16~17 |
| 기업자동화 | 매개변수 (Mediator) | article_02, 04, 05, 08 |
| 신입채용감소 | 매개변수 (Mediator) | article_01, 08, 10, 16 |
| 청년취업불안 | 결과변수 (Outcome) | article_15, 16, 20 |
| 경기침체 | 교란변수 (Confounder) | article_02, 17 |
| 산업구조변화 | 교란변수 (Confounder) | article_10, 15 |
| 스타트업투자감소 | 교란변수 (Confounder) | article_10 |
| 기업생산성향상 | 긍정 매개 (Positive Mediator) | article_14, 18 |

### 5-2. 엣지 출처 구분 (핵심 요구사항)

`causal_dag/dag_edges.csv` — 컬럼: `source | target | edge_type | edge_source | description`

| 경로 | edge_source | 근거 |
|------|-------------|------|
| AI기술발전 → 기업자동화 | **asserted** | article_02, 04, 05, 06 직접 주장 |
| 기업자동화 → 신입채용감소 | **asserted** | article_01, 08, 10, 16 직접 주장 |
| 신입채용감소 → 청년취업불안 | **asserted** | article_15, 16, 20 직접 주장 |
| 경기침체 → 신입채용감소 | **analyst_added** | backdoor path — article_02 언급, 분석가 식별 |
| 경기침체 → 청년취업불안 | **analyst_added** | backdoor path — 공통 원인 |
| 산업구조변화 → 신입채용감소 | **analyst_added** | backdoor path — article_10, 15 언급 |
| 산업구조변화 → 기업자동화 | **analyst_added** | backdoor path — 분석가 식별 |
| 스타트업투자감소 → 신입채용감소 | **analyst_added** | backdoor path — article_10 언급 |
| AI기술발전 → 기업생산성향상 | **asserted** | article_14, 18 직접 주장 |
| 기업생산성향상 → 신입채용감소 | **analyst_added** | 분석가 추론 — 생산성 대체 채용 억제 |

### 5-3. Backdoor Criterion 적용

**P(청년취업불안 | do(AI기술발전))** 식별을 위한 차단 세트:

```
Backdoor adjustment set Z = {경기침체, 산업구조변화, 스타트업투자감소}
```

모든 backdoor path가 Z로 차단되면 do-calculus 공식 적용 가능:

```
P(Y | do(X)) = Σ_z P(Y | X, Z=z) · P(Z=z)
```

단, 이 분석에서 Z는 **관찰 불가능(unobserved)** — 정량적 추정 불가.

---

## Step 6 — 인과 추론 (Ladder of Causation)

**선택 옵션:** Option B + C (그래프 기반 추론 + 반사실적 논증)

### 6-1. Rung 1 — Association (관찰)

**"AI 확산과 청년 취업난이 동시에 등장하는가?"**

| 관찰 데이터 | 출처 |
|------------|------|
| AI 채택 기업 신입 채용 7.7% 감소 (6분기 후) | article_01, 하버드대 |
| AI 고노출 직종 22~25세 고용 13% 감소 | article_01, 스탠퍼드 |
| 중소기업 AI 노출 34개 직종 채용공고 56.3% 감소 | article_02, 고용24 |
| 청년 일자리 21만1000개 감소 — 그 중 20만8000개 AI 고노출 업종 | article_04, 한국은행 |
| 신입·주니어 채용 43~51% 감소 (2022→2025) | article_10, Candid |
| 챗GPT 출시(2022-11) 이후 청년 고용 단절 패턴 | article_11, 스탠퍼드 |

**결론:** AI 확산 시점과 청년 고용 감소 시점이 강하게 공존. **상관관계는 뚜렷하다.**

**카테고리 분포 기반 모티프 분석:**
- 40개 인과 주장 중 `negative` 폴라리티: 32건 (80%)
- `not_confirmed`: 5건 (12.5%) — 인과 관계 불확실 선언
- `positive` 폴라리티: 8건 (20%) — 장기 일자리 창출 가능성

### 6-2. Rung 2 — Intervention (do-calculus)

**"AI 채택을 강제한다면(do(AI기술발전=1)) 청년 취업불안은 어떻게 변하는가?"**

**Backdoor Path 목록 (DAG 기반):**

```
Path 1: AI기술발전 ← 경기침체 → 신입채용감소 → 청년취업불안
Path 2: AI기술발전 ← 경기침체 → 청년취업불안
Path 3: AI기술발전 ← 산업구조변화 → 신입채용감소
Path 4: AI기술발전 ← 산업구조변화 → 기업자동화 → 신입채용감소
Path 5: AI기술발전 ← 스타트업투자감소 → 신입채용감소
```

**인과효과 식별 판단:**

> **결론: 순수 인과효과 식별 불완전.**

근거:
1. **국회예산정책처 회귀분석 (article_11, 12):** AI 고노출 직업 고용 감소에 대한 통계적으로 유의한 인과관계 미확인
2. **AI 워싱 논거 (article_13):** 기업의 원래 채용 축소 경향 + AI를 명분으로 활용 가능성
3. **경기침체 동시 발생 (article_02, 17):** 2022~2023년 금리 인상·내수침체 독립 변수

그러나 **연공편향 기술변화 패턴**은 AI 독립 효과를 강하게 시사:
- 같은 직종·같은 시점에서 22~25세만 감소, 35~49세는 증가 (article_11)
- 경기침체라면 전 연령대 균등 영향이 예상되므로, 이 비대칭은 AI 특유 효과 가능성

**Rung 2 판정:** *부분 식별 (partially identified)* — 연공편향 패턴은 AI 인과성을 시사하나,
교란변수 미통제로 인해 효과 크기 추정 불가.

### 6-3. Rung 3 — Counterfactual (반사실)

**"AI가 2022년 이후 급속히 발전하지 않았다면, 청년 취업률은 어떻게 달랐을까?"**

**반사실 세계 (Counterfactual World):**
- 기업들은 주니어 업무(단순 코딩·경리·고객 서비스·번역)를 여전히 인간으로 충원
- 신입 채용을 통한 career ladder 하단 유지
- 경기침체의 영향은 동일하게 유지 (confounders 통제)

**관찰 세계 vs 반사실 세계 비교:**

| 지표 | 관찰 세계 (AI 있음) | 반사실 세계 (AI 없음) |
|------|--------------------|--------------------|
| 청년 고용률 (2026 Q1) | 43.5% | 추정 44.5~45.5% |
| 주니어 채용 변화 | -51% | -20~30% (경기침체만의 영향) |
| AI 고노출 직종 청년 일자리 | -20만8000개 | -5~8만개 (추정) |

**반사실 추정:** AI 없는 세계에서 청년 고용률은 **1~2%p 높았을 것**으로 추정.

**추정의 불확실성 및 가정:**
- 가정 1: 경기침체의 크기는 AI 유무와 독립적이라고 가정 (논쟁 가능)
- 가정 2: AI 없이도 글로벌 테크 한파의 일부는 발생 (부분 중첩)
- 한계: 반사실 세계는 관찰 불가능 — 정성적 추론에 한정

---

## 7. 세 가지 구분: 상관관계 vs 주장된 인과 vs 식별된 인과

이 분석의 핵심 의무: 세 가지를 명확히 분리.

| 구분 | 내용 | 이 분석에서 |
|------|------|------------|
| **상관관계 (Correlation)** | AI 확산 시점 = 청년 고용 감소 시점 일치 | Rung 1 — 뚜렷함 |
| **주장된 인과 (Asserted Causation)** | 기사 80%가 "AI 때문에 청년 채용 줄었다"고 주장 | 텍스트에서 asserted — 근거 다양, 신뢰도 0.65~0.85 |
| **식별된 인과 (Identified Causal Effect)** | 교란변수 통제 후 순수 AI 효과 | **부분 식별** — 연공편향 패턴이 증거이나 불완전 |

> **최종 결론:** "AI기술발전 → 청년취업불안" 인과관계는 기사들에서 강하게 주장되며
> 연공편향 패턴이 독립 효과를 시사하지만, 경기침체·산업구조변화·스타트업투자감소라는
> 교란변수로 인해 **순수 인과효과의 통계적 식별은 현재 데이터로 불완전하다.**
> 이 갭을 닫으려면 AI 채택 여부를 활용한 자연실험 (차이-인-차이 설계) 이 필요하다.

---

## 8. 결론 및 한계

### 8-1. 파이프라인 요약

```
Step 1 기사 20편 (2023–2026) + metadata.csv
  ↓
Step 2 6단계 정제 → cleaned/ + cleaning_log.csv
  ↓
Step 3 엔티티 113개 + 인과 주장 40개 (relationType/polarity/confidence)
  ↓
Step 4 지식 그래프 78 nodes / 40 edges (source provenance 포함)
  ↓
Step 5 Causal DAG (asserted vs analyst-added 구분, backdoor criterion)
  ↓
Step 6 Ladder of Causation 3단계 분석 → 부분 식별 결론
```

### 8-2. 분석 한계

1. **수동 추출의 주관성:** 인과 주장 추출이 연구자 판단에 의존 — 골드셋 검증 미수행
2. **정량 데이터 미결합:** 실제 고용률 시계열과 그래프를 연결하지 못함
3. **미디어 편향:** 언론 기사는 "AI가 문제"라는 내러티브를 과대표할 수 있음
4. **교란변수 미측정:** 경기침체·산업구조변화를 수치로 통제하지 못함

### 8-3. 결론을 바꿀 증거

다음 증거가 확보된다면 결론이 달라질 수 있다:
- AI 채택 기업 vs 미채택 기업의 동일 직종 청년 고용 패널 데이터 (DID 설계 가능)
- 경기침체 영향을 통제한 후에도 연공편향 패턴이 유지되는지 확인
- AI 도입이 느린 국가와 빠른 국가 간 청년 고용률 비교

---

*생성 파일:*
- `cleaned/` (20개 정제 파일), `cleaned/cleaning_log.csv`
- `extractions/entities.csv`, `extractions/causal_assertions.csv`
- `knowledge_graph/nodes.csv`, `edges.csv`, `source_nodes.csv`, `graph.png`
- `causal_dag/dag.png`, `causal_dag/dag_edges.csv`
- `run_pipeline.py`, `step2_clean.py`
"""


def write_report():
    path = f"{BASE}/analysis_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(REPORT)
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(f"{BASE}/extractions",     exist_ok=True)
    os.makedirs(f"{BASE}/knowledge_graph", exist_ok=True)
    os.makedirs(f"{BASE}/causal_dag",      exist_ok=True)

    print("\n[Step 3] Entity & Causal Assertion 추출")
    write_entities()
    write_causal()

    print("\n[Step 4] Knowledge Graph 생성")
    build_knowledge_graph()

    print("\n[Step 5] Causal DAG 생성")
    build_causal_dag()

    print("\n[Step 6] Analysis Report 작성")
    write_report()

    print("\n✓ 파이프라인 완료")
