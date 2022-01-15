import click
import pandas as pd
import torch
from omegaconf import OmegaConf


@click.command()
@click.argument("input_filepath", type=click.Path(), default="data/raw")
@click.argument("output_filepath", type=click.Path(), default="data/processed")
def make_dataset(input_filepath: str, output_filepath: str) -> None:
    """
    Turns raw csv file into a numpy array with the two columns (title, abstract)
    """

    # get configuration
    config = OmegaConf.load("src/data/config.yaml")

    # Read csv file into pandas (only specified columns)
    data = pd.read_csv(
        f"{input_filepath}/{config.data_name}",
        usecols=[config.title_column, config.abstract_column],
    )
    data = data.to_numpy()
    torch.save(data, output_filepath + "/data.pt")


if __name__ == "__main__":
    make_dataset()
