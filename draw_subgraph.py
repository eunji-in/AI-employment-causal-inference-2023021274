"""
PPT용 서브그래프 — 노드를 텍스트 길이에 맞는 박스로 그리기
"""
import koreanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

OUT   = "/home/eunji/project/knowledge_graph/subgraph_ppt.png"
KFONT = "NanumGothic"

# ── 노드 (라벨, 레이어, 역할) ─────────────────────────────────────
NODES = [
    ("생성형AI 채택",          0, "cause"),
    ("AI 주니어 업무 대체",    0, "cause"),
    ("초급업무 AI 대체",       0, "cause"),
    ("AI 고노출 업종 확산",    0, "cause"),
    ("AI 도입+비용절감 압박",  0, "cause"),
    ("대기업 대규모 감원",     1, "med"),
    ("신입 고용 감소",         2, "out"),
    ("주니어 채용 51% 감소",   2, "out"),
    ("신입채용 감소\nZ세대 불안", 2, "out"),
    ("청년 일자리\n21만개 감소", 2, "out"),
    ("취업사다리 소멸",        2, "out"),
    ("경기침체",               3, "conf"),
    ("스타트업 투자 감소",     3, "conf"),
]

EDGES = [
    ("생성형AI 채택",        "신입 고용 감소",          "neg"),
    ("AI 주니어 업무 대체",  "주니어 채용 51% 감소",    "neg"),
    ("초급업무 AI 대체",     "신입채용 감소\nZ세대 불안","neg"),
    ("AI 고노출 업종 확산",  "청년 일자리\n21만개 감소","neg"),
    ("AI 도입+비용절감 압박","대기업 대규모 감원",       "neg"),
    ("대기업 대규모 감원",   "취업사다리 소멸",          "neg"),
    ("경기침체",             "신입 고용 감소",           "neg"),
    ("스타트업 투자 감소",   "신입 고용 감소",           "neg"),
]

role_color = {
    "cause": "#1B4F72",
    "conf":  "#7D6608",
    "med":   "#4A235A",
    "out":   "#922B21",
}
arr_color = {"neg": "#C0392B", "pos": "#1E8449"}

# ── 그래프 ────────────────────────────────────────────────────────
G = nx.DiGraph()
for lbl, lay, role in NODES:
    G.add_node(lbl, layer=lay, role=role)
for s, t, pol in EDGES:
    G.add_edge(s, t, pol=pol)

# ── 노드 위치 ─────────────────────────────────────────────────────
rng = np.random.default_rng(42)
X = {0: 0.0, 1: 3.5, 2: 7.0, 3: 3.0}   # 레이어 X 좌표

layer_nodes = {0: [], 1: [], 2: [], 3: []}
for lbl, lay, role in NODES:
    layer_nodes[lay].append(lbl)

pos = {}
for lay, nodes in layer_nodes.items():
    n = len(nodes)
    for i, lbl in enumerate(nodes):
        y = (n - 1) / 2.0 - i
        jx = rng.uniform(-0.12, 0.12) if lay != 1 else 0
        jy = rng.uniform(-0.06, 0.06)
        # 교란변수는 아래쪽에 배치
        base_y = y * 1.2 if lay != 3 else y * 1.0 - 4.5
        pos[lbl] = (X[lay] + jx, base_y + jy)

# ── 박스 크기 계산 (텍스트 길이 기반) ────────────────────────────
def box_size(lbl):
    lines = lbl.split("\n")
    max_chars = max(len(l) for l in lines)
    n_lines   = len(lines)
    w = max(max_chars * 0.18 + 0.3, 1.6)
    h = n_lines * 0.38 + 0.22
    return w, h

# ── 그리기 ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 11))
ax.set_facecolor("#F5F7FA")
fig.patch.set_facecolor("#F5F7FA")
ax.axis("off")

# 교란변수 구분선
ax.axhline(-3.2, color="#B7950B", lw=1.0, ls="--", alpha=0.55)
ax.text(8.5, -3.35, "교란변수 (Confounders) ↓",
        fontsize=8.5, color="#9A7D0A", fontstyle="italic",
        fontfamily=KFONT, ha="right", va="top")

# 엣지 — 노드 박스 경계에서 출발/도착하도록 invisible node로 먼저 그리기
# (networkx 엣지는 pos 중심 기준으로 그려짐 → min_margin으로 조정)
for s, t, pol in EDGES:
    ws, hs = box_size(s)
    wt, ht = box_size(t)
    margin_s = max(ws, hs) * 28   # pixel 단위 근사
    margin_t = max(wt, ht) * 28
    sx, sy = pos[s]; tx, ty = pos[t]
    is_conf = G.nodes[s]["role"] == "conf"
    rad = 0.25 if is_conf else 0.05
    ax.annotate(
        "", xy=(tx, ty), xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="-|>",
            color=arr_color[pol],
            lw=1.6,
            mutation_scale=10,
            shrinkA=max(ws, hs) * 38,
            shrinkB=max(wt, ht) * 38,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=2,
    )

# 노드 박스 + 라벨
for lbl, lay, role in NODES:
    x, y = pos[lbl]
    w, h = box_size(lbl)
    fc = role_color[role]
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.06",
        facecolor=fc, edgecolor="white",
        linewidth=1.4, zorder=3, alpha=0.93)
    ax.add_patch(rect)
    ax.text(x, y, lbl,
            ha="center", va="center",
            fontsize=9.0, fontweight="bold",
            color="white", fontfamily=KFONT,
            multialignment="center", zorder=4,
            linespacing=1.4)

# 컬럼 헤더
for xv, txt, fc in [(0.0, "원인", "#1B4F72"),
                     (3.5, "매개", "#4A235A"),
                     (7.0, "결과", "#922B21")]:
    ax.text(xv, 3.8, txt, ha="center",
            fontsize=11.5, fontweight="bold",
            color="white", fontfamily=KFONT,
            bbox=dict(boxstyle="round,pad=0.35",
                      fc=fc, ec="none"), zorder=5)

# 범례
leg = [
    mpatches.Patch(color="#1B4F72", label="원인 (AI)"),
    mpatches.Patch(color="#7D6608", label="교란변수"),
    mpatches.Patch(color="#4A235A", label="매개변수"),
    mpatches.Patch(color="#922B21", label="결과"),
    mpatches.Patch(color="#C0392B", label="부정 인과"),
]
ax.legend(handles=leg, loc="lower right", fontsize=9,
          framealpha=0.88, title="유형", title_fontsize=9.5,
          prop={"family": KFONT})

ax.set_title(
    "AI기술발전 → 청년취업불안  핵심 인과 경로\n"
    "(Knowledge Graph 서브그래프 · 8개 인과관계 · 기사 20편)",
    fontsize=13, fontweight="bold", pad=14,
    color="#1B4F72", fontfamily=KFONT)

# 축 범위 자동 조정
all_x = [p[0] for p in pos.values()]
all_y = [p[1] for p in pos.values()]
ax.set_xlim(min(all_x) - 1.5, max(all_x) + 1.5)
ax.set_ylim(min(all_y) - 1.0, max(all_y) + 1.0)

plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches="tight")
plt.close()
print(f"→ 저장: {OUT}")
