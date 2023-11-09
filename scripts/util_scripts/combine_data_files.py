import click
import pandas as pd
from pathlib import Path
from natsort import natsorted


@click.command()
@click.option("--data_dir", type=click.Path(exists=True,file_okay=False, path_type=Path),required=True,
               help="Directory containing the data")
def merge_data(data_dir):

    data_files = []
    files = natsorted(list(data_dir.glob("record_*.csv")))
    for f in files:
        df = pd.read_csv(f)
        df = df.dropna()
        df = df.reset_index(drop=True)
        data_files.append(df)

    final_df = pd.concat(data_files)
    final_df = final_df.reset_index(drop=True)
    final_df.to_csv(data_dir / "combined_data.csv", index=False)

if __name__ == "__main__":
    merge_data()


