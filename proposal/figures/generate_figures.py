from html import escape
from pathlib import Path


OUT = Path(__file__).parent

# SVG dimensions are expressed in millimetres, so font sizes and line widths
# are converted from points to physical millimetres.
TITLE = 9.5 * 25.4 / 72
BODY = 8.5 * 25.4 / 72
SMALL = 7.5 * 25.4 / 72
BORDER = 0.9 * 25.4 / 72
MAIN_ARROW = 1.1 * 25.4 / 72
SOFT_ARROW = 0.8 * 25.4 / 72

INK = "#202529"
MUTED = "#565f68"
BORDER_GRAY = "#555555"
BLUE_FILL = "#f5f8fa"
GREEN_FILL = "#f3f8f4"
ORANGE_FILL = "#fff7ec"
PURPLE_FILL = "#f7f4f8"
BLUE_CELL = "#dce7f0"
GREEN_CELL = "#dcebe0"
MASK_CELL = "#b8bec4"
GREEN_DARK = "#4b7057"
GREEN_TEXT = "#3f5d4a"
BLUE_TEXT = "#2f6f8f"


def esc(value):
    return escape(str(value), quote=True)


def svg(width, height, title, desc, body):
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{esc(title)}</title>
  <desc id="desc">{esc(desc)}</desc>
  <defs>
    <marker id="arrow" markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto">
      <path d="M0,0 L6,3.5 L0,7 Z" fill="#333333"/>
    </marker>
    <marker id="arrow-soft" markerWidth="6" markerHeight="6" refX="4.5" refY="3" orient="auto">
      <path d="M0,0 L5,3 L0,6 Z" fill="#4b7057"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
{body}
</svg>
'''


def box(x, y, w, h, fill, rounded=1.1, stroke=BORDER_GRAY, dashed=False, width=BORDER):
    dash = ' stroke-dasharray="1.6 1.1"' if dashed else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'rx="{rounded}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.3f}"{dash}/>'
    )


def line(x1, y1, x2, y2, width=MAIN_ARROW, color="#333333", marker="arrow"):
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width:.3f}" marker-end="url(#{marker})"/>'
    )


def text(x, y, value, size=BODY, weight="400", anchor="middle", fill=INK,
         font="Arial, Helvetica, sans-serif", italic=False):
    style = ' font-style="italic"' if italic else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="{font}" font-size="{size:.3f}" font-weight="{weight}" '
        f'fill="{fill}"{style}>{esc(value)}</text>'
    )


def serif(x, y, value, size=BODY, weight="400", anchor="middle", fill=INK):
    return text(
        x, y, value, size, weight, anchor, fill,
        font="Times New Roman, Times, serif", italic=True,
    )


def grid(x, y, rows, cols, cell_w, cell_h, masked=None, fill=BLUE_CELL,
         stroke="#8d99a6", stroke_width=0.18):
    masked = masked or set()
    out = []
    for r in range(rows):
        for c in range(cols):
            color = MASK_CELL if (r, c) in masked else fill
            out.append(
                f'<rect x="{x + c * cell_w:.2f}" y="{y + r * cell_h:.2f}" '
                f'width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{color}" '
                f'stroke="{stroke}" stroke-width="{stroke_width:.3f}"/>'
            )
    return "\n".join(out)


def cell(x, y, size, fill, stroke=GREEN_DARK, stroke_width=0.25):
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{size:.2f}" height="{size:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.3f}"/>'
    )


def stage_box(x, y, w, h, title_lines, keywords, fill, works=None):
    out = [box(x, y, w, h, fill)]
    cx = x + w / 2
    out.append(text(cx, y + 5.2, title_lines[0], TITLE, "700"))
    out.append(text(cx, y + 9.9, title_lines[1], TITLE, "700"))
    ty = y + 16.5
    if works:
        out.append(text(cx, ty, works, SMALL, "400", fill=MUTED))
        ty += 5.0
    for keyword in keywords:
        out.append(text(cx, ty, keyword, BODY))
        ty += 4.6
    return "\n".join(out)


def fig1():
    body = []
    stage_title = 10.0 * 25.4 / 72
    sub_title = 9.0 * 25.4 / 72
    model = 8.0 * 25.4 / 72

    stage_y, stage_h = 3, 47
    body.append(box(1, stage_y, 38, stage_h, "#f3f6f8"))
    body.append(text(20, 9, "I. 前期技术积累", stage_title, "700"))
    body.append(text(20, 14, "Early Technical Foundations", model, "400", fill=MUTED))
    body.append(text(20, 26, "Wireless Task Modeling", BODY))
    body.append(text(20, 31, "Transformer / SSL", BODY))
    body.append(text(20, 36, "Masked Modeling", BODY))

    body.append(line(39.2, 27.5, 44.8, 27.5, MAIN_ARROW))

    body.append(box(45, stage_y, 42, stage_h, "#eaf2f8"))
    body.append(text(66, 9, "II. 基础模型概念形成", stage_title, "700"))
    body.append(text(66, 14, "Foundation Models", model, "400", fill=MUTED))
    body.append(text(66, 26, "LWM · WiFo", BODY))
    body.append(text(66, 31, "Large-Scale Pretraining", BODY))
    body.append(text(66, 36, "Multi-Task Transfer", BODY))

    body.append(line(87.2, 27.5, 92.8, 27.5, MAIN_ARROW))

    body.append(box(93, stage_y, 86, stage_h, "#e9f5f4"))
    body.append(text(136, 9, "III. 快速扩展与专门化", stage_title, "700"))
    body.append(text(136, 14, "Expansion & Specialization", model, "400", fill=MUTED))

    branch_x = [95, 123, 151]
    branch_w = 26
    branches = [
        (("Heterogeneity", "& Scaling"), ("Heterogeneous CSI", "Generalization")),
        (("Efficiency", ""), ("Lightweight Models", "Inference Cost")),
        (("Physics /", "Structure"), ("Physical Coords", "Geometry")),
    ]
    for index, (titles, keywords) in enumerate(branches):
        x = branch_x[index]
        emphasized = index == 2
        body.append(box(
            x, 18, branch_w, 24,
            "#e5f2f1" if emphasized else "#ffffff",
            rounded=0.8,
            stroke="#3f7d82" if emphasized else "#7b8794",
            width=0.8 * 25.4 / 72,
        ))
        cx = x + branch_w / 2
        body.append(text(cx, 22.5, titles[0], sub_title, "700"))
        if titles[1]:
            body.append(text(cx, 27, titles[1], sub_title, "700"))
        keyword_size = 8.0 * 25.4 / 72
        body.append(text(cx, 33, keywords[0], keyword_size))
        body.append(text(cx, 38, keywords[1], keyword_size))

    body.append(line(164, 42, 164, 43.5, SOFT_ARROW, "#3f7d82", "arrow-soft"))
    body.append(box(149, 43.5, 30, 6.5, "#f3faf7", rounded=1.0,
                    stroke="#3f7d82", dashed=True, width=0.7 * 25.4 / 72))
    body.append(text(164, 45.9, "Focus", SMALL, "700", fill="#285b60"))
    body.append(text(164, 48.5, "Explicit 2D UPA", SMALL, "700", fill="#285b60"))

    return svg(
        180, 51,
        "Evolution of wireless foundation models",
        "A three-stage evolution from technical foundations to channel foundation models and specialized physics-aware research.",
        "\n".join(body),
    )


def fig2():
    body = [box(1, 1, 86, 52, BLUE_FILL), box(93, 1, 86, 52, GREEN_FILL)]

    body.append(text(44, 7, "Flattened Representation", TITLE, "700"))
    body.append(text(44, 12.2, "4 × 8 physical UPA", BODY, fill=MUTED))
    body.append(grid(27, 16, 4, 8, 4.0, 4.0))
    body.append(text(62.5, 25, "4 rows × 8 columns", SMALL, anchor="start", fill=MUTED))
    body.append(line(44, 35, 44, 41, MAIN_ARROW))
    body.append(text(47, 40, "Flatten", SMALL, anchor="start"))
    body.append(serif(44, 46, "u = 0, 1, ..., 31"))
    body.append(serif(44, 51, "u = rN_v + c"))

    body.append(text(136, 7, "Explicit UPA Representation", TITLE, "700"))
    body.append(text(136, 12.2, "4 × 8 row-column topology", BODY, fill=MUTED))
    body.append(grid(119, 16, 4, 8, 4.0, 4.0, fill=GREEN_CELL))

    query_x, query_y = 135, 25
    key_left_x, key_left_y = 131, 25
    key_top_x, key_top_y = 135, 21
    body.append(cell(key_left_x, key_left_y, 4, GREEN_CELL))
    body.append(cell(key_top_x, key_top_y, 4, GREEN_CELL))
    body.append(cell(query_x, query_y, 4, "#b8d4c2", stroke_width=0.30))
    body.append(text(140.5, 21.0, "key", SMALL, anchor="start", fill=GREEN_TEXT))
    body.append(text(126.5, 29.5, "key", SMALL, fill=GREEN_TEXT))
    body.append(text(140.5, 26.8, "query", SMALL, anchor="start", fill=GREEN_TEXT))
    body.append(line(query_x + 2, query_y + 2, key_left_x + 2, key_left_y + 2,
                    SOFT_ARROW, GREEN_DARK, "arrow-soft"))
    body.append(line(query_x + 2, query_y + 2, key_top_x + 2, key_top_y + 2,
                    SOFT_ARROW, GREEN_DARK, "arrow-soft"))
    body.append(text(129.2, 24.2, "Δc", SMALL, fill=GREEN_DARK))
    body.append(text(136.4, 20.2, "Δr", SMALL, fill=GREEN_DARK))
    body.append(serif(136, 41, "(t, k, u) vs. (t, k, r, c)"))
    body.append(text(136, 46, "r = row, c = column", SMALL, fill=GREEN_TEXT))
    body.append(text(136, 51, "Explicit row-column topology", SMALL, fill=GREEN_TEXT))

    body.append(text(
        44, 60,
        "Row-column topology is not explicitly encoded in the model interface",
        SMALL, fill=MUTED,
    ))
    body.append(text(136, 60, "UPA geometry is explicitly represented", SMALL, fill=GREEN_TEXT))
    return svg(
        180, 64,
        "Flattened antenna versus explicit UPA representation",
        "A conceptual contrast between one-dimensional antenna indexing and two-dimensional row-column coordinates.",
        "\n".join(body),
    )


def fig3():
    source = OUT / "upa_framework.svg"
    target = OUT / "fig3_upa_framework.svg"
    content = source.read_text()
    target.write_text(content)


def fig4():
    body = [box(1, 1, 86, 50, GREEN_FILL), box(93, 1, 86, 50, ORANGE_FILL)]

    body.append(text(44, 7, "UPA Relative Geometry", TITLE, "700"))
    grid_x, grid_y, cell_size = 20, 13, 6.0
    body.append(grid(grid_x, grid_y, 5, 5, cell_size, cell_size, fill=GREEN_CELL))

    query_x = grid_x + 2 * cell_size
    query_y = grid_y + 2 * cell_size
    key_horizontal_x = grid_x + 1 * cell_size
    key_horizontal_y = query_y
    key_vertical_x = query_x
    key_vertical_y = grid_y + 1 * cell_size
    key_diagonal_x = key_horizontal_x
    key_diagonal_y = key_vertical_y

    body.append(cell(key_horizontal_x, key_horizontal_y, cell_size, GREEN_CELL))
    body.append(cell(key_vertical_x, key_vertical_y, cell_size, GREEN_CELL))
    body.append(cell(key_diagonal_x, key_diagonal_y, cell_size, GREEN_CELL))
    body.append(cell(query_x, query_y, cell_size, "#b8d4c2", stroke_width=0.30))

    body.append(text(query_x + 3, query_y + 3.7, "i", SMALL, "700", fill=GREEN_TEXT))
    body.append(text(query_x - 3, query_y + 3.7, "j2", SMALL, "700", fill=GREEN_TEXT))
    body.append(text(query_x + 3, query_y - 2.3, "j1", SMALL, "700", fill=GREEN_TEXT))
    body.append(text(query_x - 3, query_y - 2.3, "j3", SMALL, "700", fill=GREEN_TEXT))

    body.append(line(query_x + 4.5, query_y + 4.5, key_horizontal_x + 4.5, key_horizontal_y + 4.5,
                    SOFT_ARROW, GREEN_DARK, "arrow-soft"))
    body.append(line(query_x + 4.5, query_y + 4.5, key_vertical_x + 4.5, key_vertical_y + 4.5,
                    SOFT_ARROW, GREEN_DARK, "arrow-soft"))
    body.append(line(query_x + 4.5, query_y + 4.5, key_diagonal_x + 4.5, key_diagonal_y + 4.5,
                    SOFT_ARROW, GREEN_DARK, "arrow-soft"))

    legend_x = 55
    body.append(text(legend_x, 20, "Key j1: (Δr,Δc)=(0,+1)", BODY, anchor="start", fill=GREEN_DARK))
    body.append(text(legend_x, 27, "Key j2: (Δr,Δc)=(+1,0)", BODY, anchor="start", fill=GREEN_DARK))
    body.append(text(legend_x, 34, "Key j3: (Δr,Δc)=(+1,+1)", BODY, anchor="start", fill=GREEN_DARK))
    body.append(text(44, 45, "Horizontal, vertical, and diagonal relations remain distinguishable", SMALL, fill=MUTED))

    body.append(text(136, 7, "Separable Relative Bias", TITLE, "700"))

    formula_cards = [
        (12, "Relative Displacement", "Δr = r_i - r_j;  Δc = c_i - c_j"),
        (26, "Separable Bias", "B_h(Δr,Δc) = b_r^h(Δr) + b_c^h(Δc)"),
        (40, "Attention Score", "QK^T / √d + B_h(Δr,Δc)"),
    ]
    for y, card_title, formula in formula_cards:
        body.append(box(101, y, 70, 10, "#ffffff", rounded=0.9,
                        stroke="#c7a575", width=0.8 * 25.4 / 72))
        body.append(text(136, y + 4.5, card_title, BODY, "700"))
        body.append(serif(136, y + 8.6, formula))
    return svg(
        180, 58,
        "Two-dimensional UPA relative geometry",
        "Query-key row and column displacement feeding a separable attention bias.",
        "\n".join(body),
    )


def fig5():
    body = []
    panels = [
        ("(a) Antenna Mask", "Random elements", {(0, 2), (1, 6), (2, 1), (3, 5)}),
        ("(b) Row Mask", "Entire row", {(1, c) for c in range(8)}),
        ("(c) Column Mask", "Entire column", {(r, 3) for r in range(4)}),
        ("(d) 2D Block Mask", "Contiguous block", {(r, c) for r in (1, 2) for c in (3, 4)}),
    ]

    x = 4
    for title, caption, masked in panels:
        body.append(box(x, 2, 40, 39, BLUE_FILL))
        body.append(text(x + 20, 7.5, title, BODY, "700"))
        body.append(grid(x + 6, 13, 4, 8, 3.5, 4.0, masked=masked,
                         fill="#ffffff", stroke="#98a3ab"))
        body.append(text(x + 20, 35.5, caption, SMALL, fill=MUTED))
        x += 44

    process_boxes = [
        (20, "Structured Missing Pattern"),
        (70, "Masked Reconstruction"),
        (120, "Partial-Array Recovery"),
    ]
    for x, label in process_boxes:
        body.append(box(x, 48, 40, 6.5, "#ffffff", rounded=1.0, stroke=BORDER_GRAY))
        body.append(text(x + 20, 52.5, label, SMALL, "700"))
    body.append(line(60.5, 51.2, 69.5, 51.2, MAIN_ARROW))
    body.append(line(110.5, 51.2, 119.5, 51.2, MAIN_ARROW))

    body.append(text(
        90, 59.0,
        "White cells are visible; gray cells are masked; all grids preserve the same UPA shape",
        SMALL, fill=MUTED,
    ))
    return svg(
        180, 62,
        "Structured UPA spatial masking strategies",
        "Antenna, row, column, and two-dimensional block masking examples.",
        "\n".join(body),
    )


def main():
    outputs = {
        "fig1_wfm_timeline.svg": fig1(),
        "fig2_flatten_vs_upa.svg": fig2(),
        "fig4_relative_geometry.svg": fig4(),
        "fig5_structured_masks.svg": fig5(),
    }
    for name, content in outputs.items():
        (OUT / name).write_text(content)
    fig3()


if __name__ == "__main__":
    main()
