"""
Plot the load and renewable generation of selected synthetic buildings, one figure per building.

By default it plots the three buildings of resilience/test_three_buildings.jl (IDS = [19, 1, 4]:
school 19, solar farm 1, office 4) on day 1, i.e. the first 96 rows, which is what that test simulates.

    python synthetic_data/plot_buildings.py                          # buildings 19, 1, 4, day 1
    python synthetic_data/plot_buildings.py -b 19 1 4 7 --days 2     # other buildings, two days
    python synthetic_data/plot_buildings.py --start-day 3 --forecast # day 3, with the forecasts
    python synthetic_data/plot_buildings.py --format png pdf --show

Columns read from cleaned_data/<id>.csv:
    actual_consumption_mean    realised load (act_cons in the Julia code)
    actual_pv_mean             realised generation (act_prod); wind power for the *_wind sites
    load_00_mean, pv_00_mean   with --forecast: the forecast made at a step for that same step

generate_synthetic_data.jl does not state units. The Julia optimiser adds these numbers straight to the
battery state of charge at every 15-min step, i.e. it treats them as energy per step.
"""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS_PER_DAY = 96  # 15-min steps

# generate_synthetic_data.jl: 10 archetypes x 3 instances, in this order, so ids 1-3 are solar farms, ...
ARCHETYPES = ["solar farm", "office", "residential", "industrial", "warehouse pv",
              "data center", "school", "restaurant", "coastal wind", "hilltop wind"]
INSTANCES = 3
WIND = {"coastal wind", "hilltop wind"}
NO_GENERATION = {"industrial", "data center", "restaurant"}
COLUMNS = ["actual_consumption_mean", "actual_pv_mean", "load_00_mean", "pv_00_mean"]


def archetype(bid):
    return ARCHETYPES[(bid - 1) // INSTANCES]


def read_building(bid, first_row, n_rows):
    """Rows first_row .. first_row + n_rows - 1 (0-based, header excluded) of the columns in COLUMNS."""
    data = {c: [] for c in COLUMNS}
    with open(HERE / "cleaned_data" / f"{bid}.csv", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < first_row:
                continue
            if i >= first_row + n_rows:
                break
            for c in COLUMNS:
                data[c].append(float(row[c]))
    return data


def plot_building(plt, bid, data, first_row, args):
    kind = archetype(bid)
    if kind in WIND:
        gen = "wind generation"
    elif kind in NO_GENERATION:
        gen = "generation (none for this archetype)"
    else:
        gen = "PV generation"
    n = len(data["actual_consumption_mean"])
    hours = [(first_row + i) / 4 for i in range(n)]  # hours from 00:00 of day 1

    fig, ax = plt.subplots(figsize=(11, 4.2))
    # each value holds over its 15-min step, as :steppost in the Julia plots
    ax.step(hours, data["actual_consumption_mean"], where="post", color="firebrick", lw=1.8,
            label="load (actual_consumption_mean)")
    ax.step(hours, data["actual_pv_mean"], where="post", color="seagreen", lw=1.8,
            label=f"{gen} (actual_pv_mean)")
    if args.forecast:
        ax.step(hours, data["load_00_mean"], where="post", color="firebrick", lw=1, ls="--", alpha=0.7,
                label="load forecast for the same step (load_00_mean)")
        ax.step(hours, data["pv_00_mean"], where="post", color="seagreen", lw=1, ls="--", alpha=0.7,
                label="generation forecast for the same step (pv_00_mean)")

    first_day = first_row // STEPS_PER_DAY + 1
    last_day = (first_row + n - 1) // STEPS_PER_DAY + 1
    for d in range(first_day, last_day):  # day boundaries
        ax.axvline(d * 24, color="gray", lw=0.8, ls=":")
    step = 3 if last_day == first_day else 12
    ax.set_xticks(range(int(hours[0]), int(hours[-1]) + 2, step))
    ax.set_xlim(hours[0], hours[-1] + 0.25)
    days = f"day {first_day}" if last_day == first_day else f"days {first_day}-{last_day}"
    ax.set_title(f"{kind} {bid}: load and {gen}, {days}")
    ax.set_xlabel("hour (from 00:00 of day 1)")
    ax.set_ylabel("power (15-min mean)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    saved = []
    for fmt in args.format:
        path = args.outdir / f"building_{bid}_{kind.replace(' ', '_')}.{fmt}"
        fig.savefig(path, dpi=150)
        saved.append(path)
    return fig, saved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-b", "--buildings", type=int, nargs="+", default=[19, 1, 4],
                   help="building ids, 1-30 (default: 19 1 4, as in test_three_buildings.jl)")
    p.add_argument("--days", type=int, default=1, help="number of days to plot (default 1)")
    p.add_argument("--start-day", type=int, default=1, help="first day, 1-7 (default 1)")
    p.add_argument("--forecast", action="store_true", help="also draw load_00_mean and pv_00_mean")
    p.add_argument("--outdir", type=Path, default=HERE / "plots", help="output folder (default synthetic_data/plots)")
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
        if not (HERE / "cleaned_data" / f"{bid}.csv").exists():
            sys.exit(f"no file cleaned_data/{bid}.csv")
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
