

def _style_axes(fig, ax) -> None:
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=15, fontweight="bold", pad=15)

