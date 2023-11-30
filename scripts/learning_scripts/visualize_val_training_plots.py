from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import click
import os


@click.command()
@click.option("--folder", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
def plot_npy_files(folder):
    npy_files = list(folder.glob("*.npy"))

    if not npy_files:
        print("No . npy files found in this folder.")
        return

    plt.figure()

    for i, file in enumerate(npy_files):
        data = np.load(file)
        plt.plot(data, label=file.name)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_npy_files()
