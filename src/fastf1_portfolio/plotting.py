import matplotlib.pyplot as plt


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "axes.grid": True,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )
