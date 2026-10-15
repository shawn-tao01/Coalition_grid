"""
Plot the load and generation of selected synthetic buildings, one figure per building。
"""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "synthetic_data" / "cleaned_data"
STEPS_PER_DAY = 96  # 15-min steps

# generate_synthetic_data.jl: 10 archetypes x 3 instances, in this order, so ids 1-3 are solar farms, ...
ARCHETYPES = ["solar farm", "office", "residential", "industrial", "warehouse pv",
              "data center", "school", "restaurant", "coastal wind", "hilltop wind"]
INSTANCES = 3
WIND = {"coastal wind", "hilltop wind"}
COLUMNS = ["actual_consumption_mean", "actual_pv_mean", "load_00_mean", "pv_00_mean"]


def archetype(bid):
    return ARCHETYPES[(bid - 1) // INSTANCES]


def read_building(bid, first_row, n_rows):
    """Rows first_row .. first_row + n_rows - 1 (0-based, header excluded) of the columns in COLUMNS."""
    data = {c: [] for c in COLUMNS}
    with open(DATA / f"{bid}.csv", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < first_row:
                continue
            if i >= first_row + n_rows:
                break
            for c in COLUMNS:
                data[c].append(float(row[c]))
    return data


def plot_building(plt, bid, data, first_row, args):
    gen = "wind generation" if archetype(bid) in WIND else "PV generation"
    n = len(data["actual_consumption_mean"])
    hours = [(first_row + i) / 4 for i in range(n)]  # hours from 00:00 of day 1

    fig, ax = plt.subplots(figsize=(11, 3.8))
    # each value holds over its 15-min step, as :steppost in the Julia plots
    ax.step(hours, data["actual_consumption_mean"], where="post", color="firebrick", lw=1.8, label="load")
    ax.step(hours, data["actual_pv_mean"], where="post", color="seagreen", lw=1.8, label=gen)
    if args.forecast:
        ax.step(hours, data["load_00_mean"], where="post", color="firebrick", lw=1, ls="--", alpha=0.7,
                label="load forecast")
        ax.step(hours, data["pv_00_mean"], where="post", color="seagreen", lw=1, ls="--", alpha=0.7,
                label="generation forecast")
    if args.outage:  # start of step START and end of step STOP, as in test_three_buildings.jl
        start, stop = args.outage
        for i, x in enumerate([(start - 1) / 4, stop / 4]):
            ax.axvline(x, color="red", ls="--", lw=1.5, label="outage start / end" if i == 0 else None)

    first_day = first_row // STEPS_PER_DAY + 1
    last_day = (first_row + n - 1) // STEPS_PER_DAY + 1
    for d in range(first_day, last_day):  # day boundaries
        ax.axvline(d * 24, color="gray", lw=0.8, ls=":")
    step = 5 if last_day == first_day else 12
    ax.set_xticks(range(int(hours[0]), int(hours[-1]) + 2, step))
    ax.set_xlim(hours[0], hours[-1] + 0.25)
    ax.set_title(f"building {bid}")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("energy")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    saved = []
    for fmt in args.format:
        path = args.outdir / f"building_{bid}.{fmt}"
        fig.savefig(path, dpi=150)
        saved.append(path)
    return fig, saved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-b", "--buildings", type=int, nargs="+", default=[7, 19, 20],
                   help="building ids, 1-30 (default: 7 19 20)")
    p.add_argument("--days", type=int, default=1, help="number of days to plot (default 1)")
    p.add_argument("--start-day", type=int, default=1, help="first day, 1-7 (default 1)")
    p.add_argument("--forecast", action="store_true", help="also draw load_00_mean and pv_00_mean")
    p.add_argument("--outage", type=int, nargs=2, metavar=("START", "STOP"),
                   help="mark an outage from step START to step STOP (both included)")
    p.add_argument("--outdir", type=Path, default=HERE / "output", help="output folder (default resilience/output)")
    p.add_argument("--format", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    p.add_argument("--show", action="store_true", help="also open the figures")
    args = p.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")  # save only, no window needed
    import matplotlib.pyplot as plt

    first_row = (args.start_day - 1) * STEPS_PER_DAY
    n_rows = args.days * STEPS_PER_DAY
    args.outdir.mkdir(parents=True, exist_ok=True)
    for bid in args.buildings:
        if not (DATA / f"{bid}.csv").exists():
            sys.exit(f"no file {DATA / f'{bid}.csv'}")
        data = read_building(bid, first_row, n_rows)
        if not data["actual_consumption_mean"]:
            sys.exit(f"building {bid}: no rows from day {args.start_day} on")
        if len(data["actual_consumption_mean"]) < n_rows:
            print(f"building {bid}: only {len(data['actual_consumption_mean'])} of {n_rows} rows left in the file")
        fig, saved = plot_building(plt, bid, data, first_row, args)
        print("saved", ", ".join(str(s) for s in saved))
        if not args.show:
            plt.close(fig)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
