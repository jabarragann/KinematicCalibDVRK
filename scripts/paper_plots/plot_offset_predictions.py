import numpy as np

import matplotlib.pyplot as plt


def plot_offsets():
    sub_params = dict(
        top=0.93, bottom=0.09, left=0.06, right=0.98, hspace=0.12, wspace=0.12
    )
    figsize = (15.94, 5.20)
    fig, axes = plt.subplots(6, 2, sharex=True, figsize=figsize)
    fig.subplots_adjust(**sub_params)
    model_types = ["measured-setpoint", "actual-measured"]
    for idx, type in enumerate(model_types):
        poses1 = np.load(f"results/paper_plots1/poses1_jp_{type}.npy")
        poses2 = np.load(f"results/paper_plots1/poses2_jp_{type}.npy")
        poses2_pred = np.load(f"results/paper_plots1/poses2_jp_pred_{type}.npy")

        gt = poses2 - poses1
        pred = poses2_pred - poses1

        for i in range(6):
            axes[i, idx].plot(gt[:, i], label="gt offset")
            axes[i, idx].plot(pred[:, i], label="pred offset")
            axes[i, idx].grid()

            if idx == 0:
                if i != 2:
                    axes[i, idx].set_ylabel(f"q{i+1} (rad)")
                else:
                    axes[i, idx].set_ylabel(f"q{i+1} (m)")

        axes[0, idx].set_title(f"NN{idx+1}: {type}")
        axes[0, idx].legend()
        axes[5, 0].set_xlabel("Trajectory step")
        axes[5, 1].set_xlabel("Trajectory step")

    fig.align_ylabels(axes[:, 0])

    fig.savefig("results/paper_plots1/offsets_pred.png", dpi=300)
    plt.show()

    # # Get last subplot params
    # sub_params = fig.subplotpars
    # dict_str = "sub_params = dict("
    # for param in ["top", "bottom", "left", "right", "hspace", "wspace"]:
    #     dict_str = dict_str + f"{param}={getattr(sub_params, param):0.2f}, "
    # dict_str = dict_str + ")"

    # # Get figure size
    # fig_size = fig.get_size_inches()
    # fig_str = f"figsize = ({fig_size[0]:0.2f}, {fig_size[1]:0.2f})"

    # print(dict_str)
    # print(fig_str)


if __name__ == "__main__":
    plot_offsets()
