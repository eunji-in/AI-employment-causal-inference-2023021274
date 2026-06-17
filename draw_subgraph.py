"""
PPT용 서브그래프 — 고정 레이아웃 (14×11 인치)
"""
import koreanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = "/home/eunji/project/knowledge_graph/subgraph_ppt.png"

# ── 엣지 정의: (원인, 결과, 폴라리티, 그룹) ───────────────────────
EDGES = [
    # AI 직접 대체
    ("생성형AI 채택",          "신입 고용 감소",                "neg", "AI"),
    ("AI 주니어\n업무 대체",   "주니어 채용 51% 감소",          "neg", "AI"),
    ("초급 업무\nAI 대체",     "신입채용 감소\n(Z세대 취업불안)","neg", "AI"),
    ("AI 고노출\n업종 확산",   "청년 일자리\n21만개 감소",       "neg", "AI"),
    # 채용 구조 변화
    ("AI 도입 +\n비용절감 압박","대기업\n대규모 감원",           "neg", "채용"),
    ("대기업 감원 +\n신입 축소","취업사다리\n소멸",              "neg", "채용"),
    ("AI 단순업무\n자동화",     "글로벌 기업 66%\n초급 채용 축소","neg","채용"),
    # 교란변수
    ("경기침체",               "채용공고 감소",                 "neg", "conf"),
    ("스타트업 투자\n20% 감소", "2024 채용\n역대 최저",          "neg", "conf"),
    # 긍정 반론
    ("AI 생산성\n향상(장기)",   "노동 수요\n확대 가능성",         "pos", "pos"),
    ("AI 기술 채택",           "AI·ML 전문가\n수요 35%↑",       "pos", "pos"),
]

# ── 색상 ──────────────────────────────────────────────────────────
CAUSE_COLOR = {"AI": "#1A3A5C", "채용": "#1F618D",
               "conf": "#784212", "pos": "#1E8449"}
EFF_COLOR   = {"neg": "#B03A2E", "pos": "#1E8449"}
ARR_COLOR   = {"neg": "#E74C3C", "pos": "#27AE60"}
BG_COLOR    = {"AI": "#D6EAF8", "채용": "#D5D8DC",
               "conf": "#FAE5D3", "pos": "#D5F5E3"}
GRP_LABEL   = {"AI": "AI 직접 대체 경로", "채용": "채용 구조 변화",
               "conf": "교란변수 (Confounder)", "pos": "긍정 경로 (반론)"}

# ── 고정 Y 위치 계산 ──────────────────────────────────────────────
SPACING   = 1.05   # 항목 간 y 간격
GROUP_GAP = 0.60   # 그룹 간 추가 간격
ORDER     = ["AI", "채용", "conf", "pos"]

# 그룹별 항목
from collections import OrderedDict
grp_items = OrderedDict((g, []) for g in ORDER)
for cause, eff, pol, grp in EDGES:
    grp_items[grp].append((cause, eff, pol))

# y 좌표 배정
cy_map, ey_map = {}, {}
grp_spans = {}
y_cur = 0.0

for grp in ORDER:
    items = grp_items[grp]
    ys = []
    for cause, eff, pol in items:
        ys.append(-y_cur)
        cy_map[cause] = -y_cur
        ey_map[eff]   = -y_cur
        y_cur += SPACING
    grp_spans[grp] = (min(ys), max(ys))
    y_cur += GROUP_GAP

Y_MIN = -y_cur + GROUP_GAP + 0.3
Y_MAX = 1.0

# ── 그리기 ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 11))
ax.set_facecolor("#F7F9FC")
fig.patch.set_facecolor("#F7F9FC")
ax.set_xlim(-0.62, 1.62)
ax.set_ylim(Y_MIN, Y_MAX)
ax.axis("off")

NW, NH = 0.30, 0.62   # 노드 박스 크기

def draw_box(cx, cy, label, fc, fs=9.2):
    box = mpatches.FancyBboxPatch(
        (cx - NW/2, cy - NH/2), NW, NH,
        boxstyle="round,pad=0.04",
        facecolor=fc, edgecolor="white", linewidth=1.5, zorder=3, alpha=0.93)
    ax.add_patch(box)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fs, color="white", fontweight="bold",
            zorder=4, linespacing=1.3, multialignment="center")

# 그룹 배경
for grp in ORDER:
    y_top, y_bot = grp_spans[grp]
    pad = SPACING * 0.42
    rect = mpatches.FancyBboxPatch(
        (-0.58, y_bot - pad), 2.16, (y_top - y_bot + pad*2),
        boxstyle="round,pad=0.04",
        facecolor=BG_COLOR[grp], edgecolor="#BFC9CA",
        linewidth=0.8, zorder=0, alpha=0.45)
    ax.add_patch(rect)
    ax.text(-0.55, (y_top + y_bot) / 2,
            GRP_LABEL[grp], ha="left", va="center",
            fontsize=8, color="#555", fontstyle="italic", zorder=1)

# 화살표
for cause, eff, pol, grp in EDGES:
    cy = cy_map[cause]; ey = ey_map[eff]
    col = ARR_COLOR[pol]
    ax.annotate("",
        xy=(0.5 - NW/2 - 0.01, ey),
        xytext=(0.5 + NW/2 + 0.01 - 0.5, cy),  # X 조정은 아래서
        arrowprops=dict(arrowstyle="-|>", color=col,
                        lw=1.7, mutation_scale=15,
                        connectionstyle="arc3,rad=0.0"),
        zorder=2)

# 화살표 재작성 (정확한 X 사용)
# 원인 X=0.12, 결과 X=0.88
X_CAUSE = 0.12
X_EFF   = 0.88

for cause, eff, pol, grp in EDGES:
    cy_v = cy_map[cause]; ey_v = ey_map[eff]
    col = ARR_COLOR[pol]
    ax.annotate("",
        xy=(X_EFF - NW/2 - 0.01, ey_v),
        xytext=(X_CAUSE + NW/2 + 0.01, cy_v),
        arrowprops=dict(arrowstyle="-|>", color=col,
                        lw=1.7, mutation_scale=15,
                        connectionstyle="arc3,rad=0.0"),
        zorder=2)

# 노드 그리기
drawn_c, drawn_e = set(), set()
for cause, eff, pol, grp in EDGES:
    if cause not in drawn_c:
        draw_box(X_CAUSE, cy_map[cause], cause, CAUSE_COLOR[grp])
        drawn_c.add(cause)
    if eff not in drawn_e:
        draw_box(X_EFF, ey_map[eff], eff, EFF_COLOR[pol], fs=8.8)
        drawn_e.add(eff)

# 헤더
for xh, lbl, fc in [(X_CAUSE, "원인 / 메커니즘", "#1A3A5C"),
                     (X_EFF,   "결과",            "#922B21")]:
    ax.text(xh, 0.82, lbl, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec="none"), zorder=5)
ax.text(0.5, 0.82, "→", ha="center", va="center", fontsize=16, color="#666")

# 범례
leg = [
    mpatches.Patch(color="#1A3A5C", label="AI 직접 대체"),
    mpatches.Patch(color="#1F618D", label="채용 구조 변화"),
    mpatches.Patch(color="#784212", label="교란변수"),
    mpatches.Patch(color="#1E8449", label="긍정 경로 (반론)"),
    mpatches.Patch(color="#B03A2E", label="부정 결과"),
    mpatches.Patch(color="#1E8449", label="긍정 결과"),
]
ax.legend(handles=leg, loc="lower right", fontsize=9,
          framealpha=0.9, title="유형", title_fontsize=9.5)

ax.set_title(
    "AI기술발전 → 청년취업불안  핵심 인과 경로\n"
    "(Knowledge Graph 서브그래프 · 11개 인과관계 · 기사 20편)",
    fontsize=13, fontweight="bold", pad=10, color="#1A3A5C")

plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches="tight")
plt.close()
print(f"→ 저장: {OUT}")
