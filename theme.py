import matplotlib as mpl

BG     = "#0E1117"
FG     = "#E6EDF3"
GRID   = "#2F3742"
ACCENT = "#4FB3FF"
WIDGET_BG        = "#161B22"
WIDGET_BORDER    = "#30363D"
WIDGET_TEXT      = "#E6EDF3"
WIDGET_ACTIVE    = "#58A6FF"
WIDGET_HOVER     = "#1F6FEB"
WIDGET_DISABLED  = "#21262D"

HIGHLIGHT = "#F78166"
UI_BG     = "#30363D"

def apply_theme():
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "figure.edgecolor": BG,

        "axes.facecolor":   BG,
        "axes.edgecolor":   GRID,
        "axes.labelcolor":  FG,
        "axes.titlecolor":  FG,

        "xtick.color":      FG,
        "ytick.color":      FG,

        "text.color":       FG,

        "grid.color":       GRID,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",

        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False
    })