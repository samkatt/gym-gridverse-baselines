"""Entrypoint of baseline experiments result plotting

Functions as a gateway to visualizing results. Accepts a 'type' of plot
followed by a list of file names. For example::

    python plot_results.py mean_bar /path/to/file1.pkl /path/to/file2.pkl /path/to/file3.pkl

"""

import argparse
from typing import Any, Dict
import matplotlib.pyplot as plt

import pandas as pd


def main():
    """Main entry point of gym-gridverse-baselines planning"""

    global_parser = argparse.ArgumentParser()
    global_parser.add_argument(
        "plot_type", choices=["mean_bar"], help="Type of plot to visualize"
    )

    global_parser.add_argument("files", nargs="+")

    args = global_parser.parse_args()

    results = {f: pd.read_pickle(open(f, "rb")) for f in args.files}

    # add discounted reward because that will be usefull to almost everyone
    for r in results.values():
        r["data"]["discounted_reward"] = (
            r["data"].reward * r["meta"]["discount_factor"] ** r["data"].timestep
        )

    if args.plot_type == "mean_bar":
        plot_mean_bars(results)
    else:
        raise ValueError(f"Unexpected plot type {args.plot_type}")

    plt.show()


def plot_mean_bars(results: Dict[str, Dict[str, Any]]):
    """Returns a (matplotlib) ax box plot

    :param results:
    :return: an `AxesSubplot`, or just do `matplotlib.pyplot.show()`
    """
    for f, r in results.items():
        r["discounted_return"] = (
            r["data"].groupby(["episode"]).agg(**{f: ("discounted_reward", "sum")})
        )

    return pd.concat(
        (r["discounted_return"] for r in results.values()), axis=1
    ).boxplot()


if __name__ == "__main__":
    main()
