# Figure Notes

All five figures are flat vector graphics with a white background and are inserted as two-column `figure*` floats. The final PDFs use Arial/Helvetica for labels and Times New Roman for formulas. Minimum text size is 7.5 pt (7.51 pt in Fig. 3 due to unit rounding).

| Figure | Purpose | Manuscript location | Layout | Source size | Minimum font | Readability check |
|---|---|---|---|---|---|---|
| Fig. 1 | Shows the three-stage evolution from early technical foundations to channel foundation models and rapid specialization, with the explicit 2D UPA focus placed under the physics/structure-awareness branch. | Section II, WFM technical evolution | Two-column | 180 × 51 mm | 8.0 pt | Pass; branch labels fit inside their boxes and the focus callout remains inside the 180 mm canvas. |
| Fig. 2 | Contrasts the flattened antenna index \(u\) with explicit row-column coordinates \((r,c)\), emphasizing model-interface explicitness rather than an impossibility claim. | Section II, UPA representation and research gap | Two-column | 180 × 64 mm | 7.5 pt | Pass; bottom annotations fit within the figure boundary. |
| Fig. 3 | Presents the complete method pipeline: shared per-antenna time-frequency embedding, \((t,k,r,c)\) tokens, UPA relative bias, and structured masked reconstruction/prediction. | Section IV, overall research method | Two-column | 179 × 68 mm | 7.51 pt | Pass; the lower output labels are split into two short lines and remain inside the rightmost module. |
| Fig. 4 | Explains horizontal, vertical, and diagonal query-key displacements and the separable row/column attention bias. | Section IV, UPA relative geometry | Two-column | 180 × 58 mm | 7.5 pt | Pass; formulas are grouped into three aligned cards and all legend labels fit inside the panel. |
| Fig. 5 | Illustrates antenna, row, column, and 2D-block spatial masks with a shared visible/masked legend. | Section IV, UPA structured spatial masking | Two-column | 180 × 62 mm | 7.5 pt | Pass; panel, process, and legend groups are centered on the 90 mm figure axis. |

Compilation was performed with `tectonic`. The final manuscript has no unresolved citations or labels and no overfull boxes. All fonts reported by `pdffonts` are embedded. A rendered-text check confirms that labels in Figs. 1-5 do not cross their enclosing boxes. The references flow from page 8 to page 9 without an intervening half-empty page. The remaining underfull-vbox notices correspond to float-only pages rather than large visual blank areas; pixel-level page checks found no half-page blank regions.
