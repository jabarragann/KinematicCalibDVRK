import matplotlib.pyplot as plt


def create_joint_plot_for_IK_FK_tests(measured_js, calculated_js):
    """Compare measured joint states versus joint states calculated with IK model"""
    sub_params = dict(
        top=0.94,
        bottom=0.08,
        left=0.06,
        right=0.95,
        hspace=0.93,
        wspace=0.20,
    )
    figsize = (9.28, 6.01)

    fig, axes = plt.subplots(6, 2, figsize=figsize, sharex=True)
    fig.subplots_adjust(**sub_params)

    for i in range(6):
        axes[i, 0].plot(measured_js[:, i], label="measured_js")
        axes[i, 0].plot(calculated_js[:, i], label="IK(measured_cp)")
        axes[i, 0].set_title(f"Joint {i+1}")

        axes[i, 1].plot(calculated_js[:, i] - measured_js[:, i], label="IK")
        axes[i, 1].set_title(f"measured_js - IK(measured_cp) for joint {i+1}")

    axes[0, 0].legend()
    plt.show()
