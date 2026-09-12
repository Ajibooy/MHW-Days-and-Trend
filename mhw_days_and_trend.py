# ==============================================================
# TROPICAL NORTH EAST ATLANTIC
# MEAN MHW DAYS AND MHW DAYS TREND
#
# Region:
#   0-30°N, 60-10°W
#
# Period:
#   1982-2024
#
# Baseline:
#   Daily P90 threshold from 1981-2010
#
# MHW definition:
#   SST > daily P90 for at least 5 consecutive days
#
# Panel A:
#   Mean MHW Days
#   Units = Days/Year
#
# Panel B:
#   MHW Days Trend
#   Units = Days/Decade
#
# Black dots in Panel B:
#   statistically significant trend (p < 0.05)
# ==============================================================


import os
import glob
import gc
import time
import warnings
import traceback

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.mpl.ticker import (
    LongitudeFormatter,
    LatitudeFormatter
)

from scipy.stats import t as student_t


warnings.filterwarnings("ignore")


# ==============================================================
# 1. USER SETTINGS
# ==============================================================

DATA_DIR = r"C:\Users\Aina Ajibola\Desktop\oisst_data"

THRESHOLD_FILE = (
    r"C:\Users\Aina Ajibola\Desktop"
    r"\P90_1981-2010\threshold.nc"
)

LAT_MIN = 0.0
LAT_MAX = 30.0

LON_MIN = -60.0
LON_MAX = -10.0

START_DATE = "1982-01-01"
END_DATE = "2024-12-31"

START_YEAR = 1982
END_YEAR = 2024

MIN_DURATION = 5

LAT_BLOCK_SIZE = 20

SIGNIFICANCE_LEVEL = 0.05

MAP_CMAP = "turbo"

USE_CHECKPOINTS = True

CHECKPOINT_DIR = (
    r"C:\Users\Aina Ajibola\Desktop"
    r"\MHW_days_checkpoints"
)


# ==============================================================
# 2. PREPROCESS DAILY OISST
# ==============================================================

def preprocess(ds):

    rename = {}

    for old, new in {
        "latitude": "lat",
        "longitude": "lon",
        "Time": "time",
        "TIME": "time"
    }.items():

        if old in ds.coords or old in ds.dims:
            rename[old] = new

    if rename:
        ds = ds.rename(rename)

    for dim in [
        "zlev",
        "depth",
        "lev",
        "level"
    ]:

        if dim in ds.dims and ds.sizes[dim] == 1:

            ds = ds.squeeze(
                dim,
                drop=True
            )

    if "sst" not in ds.data_vars:

        raise KeyError(
            "Variable 'sst' was not found."
        )

    ds = ds[["sst"]]

    # ----------------------------------------------------------
    # Convert longitude from 0-360 to -180 to 180 if necessary
    # ----------------------------------------------------------

    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(
            lon=((ds.lon + 180.0) % 360.0) - 180.0
        )

    ds = ds.sortby("lon")
    ds = ds.sortby("lat")

    ds = ds.sel(
        lat=slice(
            LAT_MIN,
            LAT_MAX
        ),
        lon=slice(
            LON_MIN,
            LON_MAX
        )
    )

    return ds


# ==============================================================
# 3. CLIMATOLOGICAL DAY INDEX
#
# Uses leap year 2000 as reference:
# Jan 1 = 0
# ...
# Feb 29 included
# ...
# Dec 31 = 365
# ==============================================================

def climatological_day_index(dates):

    dates = pd.DatetimeIndex(dates)

    reference_dates = pd.DatetimeIndex(
        pd.to_datetime(
            {
                "year": np.full(
                    len(dates),
                    2000
                ),
                "month": dates.month,
                "day": dates.day
            }
        )
    )

    return (
        reference_dates.dayofyear.to_numpy(
            dtype=np.int16
        )
        - 1
    )


# ==============================================================
# 4. DETECT ANNUAL MHW DAYS
#
# Important:
#
# If:
#
# Jan 1 > P90
# Jan 2 > P90
# Jan 3 > P90
# Jan 4 > P90
# Jan 5 > P90
#
# then:
#
# MHW events = 1
# MHW days   = 5
#
# This function calculates MHW DAYS, not frequency.
# ==============================================================

def detect_annual_mhw_days(
    exceedance,
    dates,
    years
):

    n_time, n_lat, n_lon = exceedance.shape

    n_years = len(years)

    n_cells = (
        n_lat
        *
        n_lon
    )

    # ----------------------------------------------------------
    # Flatten spatial dimensions
    # ----------------------------------------------------------

    hot = exceedance.reshape(
        n_time,
        n_cells
    )

    annual_days = np.zeros(
        (
            n_years,
            n_cells
        ),
        dtype=np.uint16
    )

    # ----------------------------------------------------------
    # Track current event
    # ----------------------------------------------------------

    run_length = np.zeros(
        n_cells,
        dtype=np.uint16
    )

    run_start = np.full(
        n_cells,
        -1,
        dtype=np.int32
    )

    cell_indices = np.arange(
        n_cells
    )

    # ----------------------------------------------------------
    # Loop through time
    # ----------------------------------------------------------

    for t in range(n_time):

        today_hot = hot[t]

        # ------------------------------------------------------
        # New exceedance sequence starts
        # ------------------------------------------------------

        newly_started = (
            today_hot
            &
            (run_length == 0)
        )

        run_start[
            newly_started
        ] = t

        # ------------------------------------------------------
        # Continue sequence
        # ------------------------------------------------------

        run_length[
            today_hot
        ] += 1

        # ------------------------------------------------------
        # Sequence ends
        # ------------------------------------------------------

        ended = (
            ~today_hot
            &
            (run_length > 0)
        )

        qualifying = (
            ended
            &
            (run_length >= MIN_DURATION)
        )

        # ------------------------------------------------------
        # Process completed MHW events
        # ------------------------------------------------------

        qualifying_cells = cell_indices[
            qualifying
        ]

        for cell in qualifying_cells:

            start = run_start[cell]

            end = t - 1

            # --------------------------------------------------
            # Count every day belonging to this MHW
            #
            # This correctly handles events crossing Dec-Jan.
            # --------------------------------------------------

            event_dates = dates[
                start:end + 1
            ]

            event_years = event_dates.year

            unique_years, counts = np.unique(
                event_years,
                return_counts=True
            )

            for event_year, count in zip(
                unique_years,
                counts
            ):

                if (
                    START_YEAR
                    <= event_year
                    <= END_YEAR
                ):

                    year_index = (
                        event_year
                        -
                        START_YEAR
                    )

                    annual_days[
                        year_index,
                        cell
                    ] += count

        # ------------------------------------------------------
        # Reset ended runs
        # ------------------------------------------------------

        run_length[
            ended
        ] = 0

        run_start[
            ended
        ] = -1


    # ==========================================================
    # EVENT ACTIVE ON FINAL DATA DAY
    # ==========================================================

    qualifying = (
        run_length
        >=
        MIN_DURATION
    )

    qualifying_cells = cell_indices[
        qualifying
    ]

    for cell in qualifying_cells:

        start = run_start[
            cell
        ]

        end = n_time - 1

        event_dates = dates[
            start:end + 1
        ]

        event_years = event_dates.year

        unique_years, counts = np.unique(
            event_years,
            return_counts=True
        )

        for event_year, count in zip(
            unique_years,
            counts
        ):

            if (
                START_YEAR
                <= event_year
                <= END_YEAR
            ):

                year_index = (
                    event_year
                    -
                    START_YEAR
                )

                annual_days[
                    year_index,
                    cell
                ] += count


    return annual_days.reshape(
        n_years,
        n_lat,
        n_lon
    )


# ==============================================================
# 5. CALCULATE MHW DAYS TREND
# ==============================================================

def calculate_days_trend(
    annual_days,
    ocean_mask
):

    y = annual_days.astype(
        np.float64
    )

    n_years = y.shape[0]

    x = np.arange(
        START_YEAR,
        END_YEAR + 1,
        dtype=np.float64
    )

    x_mean = np.mean(x)

    x_centered = (
        x
        -
        x_mean
    )

    ss_x = np.sum(
        x_centered ** 2
    )

    # ----------------------------------------------------------
    # Mean MHW days at each grid cell
    # ----------------------------------------------------------

    y_mean = np.mean(
        y,
        axis=0
    )

    # ----------------------------------------------------------
    # Linear trend
    # ----------------------------------------------------------

    slope_year = (
        np.sum(
            x_centered[
                :,
                None,
                None
            ]
            *
            (
                y
                -
                y_mean[
                    None,
                    :,
                    :
                ]
            ),
            axis=0
        )
        /
        ss_x
    )

    # ----------------------------------------------------------
    # Convert days/year to days/decade
    # ----------------------------------------------------------

    trend_decade = (
        slope_year
        *
        10.0
    )


    # ==========================================================
    # STATISTICAL SIGNIFICANCE
    # ==========================================================

    fitted = (
        y_mean[
            None,
            :,
            :
        ]
        +
        slope_year[
            None,
            :,
            :
        ]
        *
        x_centered[
            :,
            None,
            None
        ]
    )

    residuals = (
        y
        -
        fitted
    )

    sse = np.sum(
        residuals ** 2,
        axis=0
    )

    degrees_freedom = (
        n_years
        -
        2
    )

    residual_variance = (
        sse
        /
        degrees_freedom
    )

    slope_se = np.sqrt(
        residual_variance
        /
        ss_x
    )

    t_stat = np.zeros_like(
        slope_year
    )

    good = (
        slope_se > 0
    )

    t_stat[
        good
    ] = (
        slope_year[
            good
        ]
        /
        slope_se[
            good
        ]
    )

    p_value = np.ones_like(
        slope_year
    )

    p_value[
        good
    ] = (
        2.0
        *
        student_t.sf(
            np.abs(
                t_stat[
                    good
                ]
            ),
            degrees_freedom
        )
    )


    # ----------------------------------------------------------
    # Land mask
    # ----------------------------------------------------------

    trend_decade = np.where(
        ocean_mask,
        trend_decade,
        np.nan
    )

    p_value = np.where(
        ocean_mask,
        p_value,
        np.nan
    )


    return (
        trend_decade,
        p_value
    )


# ==============================================================
# 6. CHECKPOINT
# ==============================================================

def checkpoint_path(
    block_start,
    block_end
):

    return os.path.join(
        CHECKPOINT_DIR,
        (
            f"mhw_days_lat_"
            f"{block_start:03d}_"
            f"{block_end:03d}.npz"
        )
    )


# ==============================================================
# 7. MAIN
# ==============================================================

def main():

    sst_ds = None
    threshold_ds = None

    start_clock = time.perf_counter()

    try:

        # ======================================================
        # CHECKPOINT DIRECTORY
        # ======================================================

        if USE_CHECKPOINTS:

            os.makedirs(
                CHECKPOINT_DIR,
                exist_ok=True
            )


        # ======================================================
        # FIND DAILY OISST FILES
        # ======================================================

        files = sorted(
            glob.glob(
                os.path.join(
                    DATA_DIR,
                    "*_oisst.nc"
                )
            )
        )

        if not files:

            files = sorted(
                glob.glob(
                    os.path.join(
                        DATA_DIR,
                        "*.nc"
                    )
                )
            )

        if not files:

            raise FileNotFoundError(
                f"No NetCDF files found in:\n"
                f"{DATA_DIR}"
            )


        print("=" * 85)

        print(
            "TROPICAL NORTH EAST ATLANTIC "
            "MHW DAYS ANALYSIS"
        )

        print("=" * 85)

        print(
            f"\nSST files found: "
            f"{len(files):,}"
        )


        # ======================================================
        # OPEN DAILY OISST
        # ======================================================

        print(
            "\nOpening daily OISST..."
        )

        sst_ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            preprocess=preprocess,
            parallel=False,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="outer",
            engine="netcdf4"
        )

        sst_ds = sst_ds.sortby(
            "time"
        )

        sst = sst_ds[
            "sst"
        ]


        # ======================================================
        # TIME
        # ======================================================

        time_index = pd.DatetimeIndex(
            sst.time.values
        ).normalize()

        sst = sst.assign_coords(
            time=time_index
        )

        keep = np.where(
            ~time_index.duplicated(
                keep="first"
            )
        )[0]

        sst = sst.isel(
            time=keep
        )

        sst = sst.sel(
            time=slice(
                START_DATE,
                END_DATE
            )
        )

        dates = pd.DatetimeIndex(
            sst.time.values
        )


        print(
            f"Analysis period: "
            f"{dates[0].date()} to "
            f"{dates[-1].date()}"
        )

        print(
            f"Number of days: "
            f"{len(dates):,}"
        )

        print(
            f"Grid: "
            f"{sst.sizes['lat']} × "
            f"{sst.sizes['lon']}"
        )


        # ======================================================
        # SST UNITS
        # ======================================================

        sample = float(
            sst.isel(
                time=0,
                lat=slice(
                    0,
                    min(
                        5,
                        sst.sizes["lat"]
                    )
                ),
                lon=slice(
                    0,
                    min(
                        5,
                        sst.sizes["lon"]
                    )
                )
            )
            .mean(
                skipna=True
            )
            .compute()
        )

        CONVERT_KELVIN = (
            sample > 100
        )

        if CONVERT_KELVIN:

            print(
                "SST detected as Kelvin."
            )

        else:

            print(
                "SST detected as °C."
            )


        # ======================================================
        # OPEN DAILY P90
        # ======================================================

        print(
            "\nOpening P90 threshold..."
        )

        threshold_ds = xr.open_dataset(
            THRESHOLD_FILE,
            engine="netcdf4"
        )

        if "p90" not in threshold_ds.data_vars:

            raise KeyError(
                "Variable 'p90' was not found."
            )

        p90 = threshold_ds[
            "p90"
        ].squeeze(
            drop=True
        )

        if float(
            p90.lon.max()
        ) > 180:

            p90 = p90.assign_coords(
                lon=(
                    (
                        p90.lon
                        +
                        180.0
                    )
                    %
                    360.0
                )
                -
                180.0
            )

        p90 = p90.sortby(
            "lon"
        )

        p90 = p90.sortby(
            "lat"
        )

        p90 = p90.sel(
            lat=slice(
                LAT_MIN,
                LAT_MAX
            ),
            lon=slice(
                LON_MIN,
                LON_MAX
            )
        )


        # ======================================================
        # P90 DIMENSION CHECK
        # ======================================================

        if "clim_day" not in p90.dims:

            raise ValueError(
                "P90 threshold does not contain 'clim_day'."
            )

        if p90.sizes[
            "clim_day"
        ] != 366:

            raise ValueError(
                "Expected 366 climatological days in P90."
            )


        # ======================================================
        # GRID CHECK
        # ======================================================

        if not np.allclose(
            sst.lat.values,
            p90.lat.values
        ):

            raise ValueError(
                "SST and P90 latitude grids do not match."
            )

        if not np.allclose(
            sst.lon.values,
            p90.lon.values
        ):

            raise ValueError(
                "SST and P90 longitude grids do not match."
            )

        print(
            "SST and P90 grids match."
        )


        # ======================================================
        # CLIMATOLOGICAL DAY INDEX
        # ======================================================

        clim_indices = climatological_day_index(
            dates
        )

        years = np.arange(
            START_YEAR,
            END_YEAR + 1
        )

        n_years = len(
            years
        )


        # ======================================================
        # OUTPUT ARRAYS
        # ======================================================

        n_lat = sst.sizes[
            "lat"
        ]

        n_lon = sst.sizes[
            "lon"
        ]

        annual_mhw_days = np.zeros(
            (
                n_years,
                n_lat,
                n_lon
            ),
            dtype=np.uint16
        )

        ocean_mask = np.zeros(
            (
                n_lat,
                n_lon
            ),
            dtype=bool
        )


        # ======================================================
        # BLOCK PROCESSING
        # ======================================================

        total_blocks = int(
            np.ceil(
                n_lat
                /
                LAT_BLOCK_SIZE
            )
        )

        print(
            f"\nProcessing {total_blocks} latitude blocks..."
        )


        for block_number, block_start in enumerate(
            range(
                0,
                n_lat,
                LAT_BLOCK_SIZE
            ),
            start=1
        ):

            block_end = min(
                block_start
                +
                LAT_BLOCK_SIZE,
                n_lat
            )

            checkpoint = checkpoint_path(
                block_start,
                block_end
            )

            print(
                "\n"
                + "-" * 70
            )

            print(
                f"Block {block_number}/{total_blocks} | "
                f"Latitude rows "
                f"{block_start + 1}-{block_end}"
            )


            # ==================================================
            # EXISTING CHECKPOINT
            # ==================================================

            if (
                USE_CHECKPOINTS
                and
                os.path.exists(
                    checkpoint
                )
            ):

                saved = np.load(
                    checkpoint
                )

                annual_mhw_days[
                    :,
                    block_start:block_end,
                    :
                ] = saved[
                    "annual_mhw_days"
                ]

                ocean_mask[
                    block_start:block_end,
                    :
                ] = saved[
                    "ocean_mask"
                ]

                print(
                    "Loaded from checkpoint."
                )

                continue


            # ==================================================
            # LOAD SST BLOCK
            # ==================================================

            sst_block = np.asarray(
                sst.isel(
                    lat=slice(
                        block_start,
                        block_end
                    )
                ).values,
                dtype=np.float32
            )

            if CONVERT_KELVIN:

                sst_block = (
                    sst_block
                    -
                    273.15
                )


            # ==================================================
            # LOAD P90 BLOCK
            # ==================================================

            p90_block = np.asarray(
                p90.isel(
                    lat=slice(
                        block_start,
                        block_end
                    )
                ).values,
                dtype=np.float32
            )

            # --------------------------------------------------
            # Convert 366-day climatology into threshold for
            # every actual day between 1982 and 2024.
            # --------------------------------------------------

            daily_p90 = p90_block[
                clim_indices,
                :,
                :
            ]


            # ==================================================
            # OCEAN MASK
            # ==================================================

            block_ocean = (
                np.any(
                    np.isfinite(
                        sst_block
                    ),
                    axis=0
                )
                &
                np.any(
                    np.isfinite(
                        p90_block
                    ),
                    axis=0
                )
            )

            ocean_mask[
                block_start:block_end,
                :
            ] = block_ocean


            # ==================================================
            # DAILY SST > DAILY P90
            # ==================================================

            exceedance = (
                np.isfinite(
                    sst_block
                )
                &
                np.isfinite(
                    daily_p90
                )
                &
                (
                    sst_block
                    >
                    daily_p90
                )
            )


            # ==================================================
            # DETECT MHW DAYS
            # ==================================================

            block_days = detect_annual_mhw_days(
                exceedance,
                dates,
                years
            )

            block_days[
                :,
                ~block_ocean
            ] = 0

            annual_mhw_days[
                :,
                block_start:block_end,
                :
            ] = block_days


            # ==================================================
            # SAVE CHECKPOINT
            # ==================================================

            if USE_CHECKPOINTS:

                np.savez_compressed(
                    checkpoint,
                    annual_mhw_days=
                        block_days,
                    ocean_mask=
                        block_ocean
                )


            # ==================================================
            # MEMORY CLEANUP
            # ==================================================

            del sst_block
            del p90_block
            del daily_p90
            del exceedance
            del block_days
            del block_ocean

            gc.collect()

            print(
                "Block completed."
            )


        # ======================================================
        # MEAN MHW DAYS
        # ==============================================================

        print(
            "\nCalculating mean MHW days..."
        )

        mean_mhw_days = np.mean(
            annual_mhw_days.astype(
                np.float64
            ),
            axis=0
        )

        mean_mhw_days = np.where(
            ocean_mask,
            mean_mhw_days,
            np.nan
        )


        # ======================================================
        # MHW DAYS TREND
        # ==============================================================

        print(
            "Calculating MHW days trend..."
        )

        (
            mhw_days_trend,
            trend_p_value
        ) = calculate_days_trend(
            annual_mhw_days,
            ocean_mask
        )

        significant = (
            np.isfinite(
                trend_p_value
            )
            &
            (
                trend_p_value
                <
                SIGNIFICANCE_LEVEL
            )
        )


        # ======================================================
        # MAP DATA
        # ==============================================================

        ocean_days = mean_mhw_days[
            ocean_mask
        ]

        ocean_trend = mhw_days_trend[
            ocean_mask
        ]

        lon_values = sst.lon.values

        lat_values = sst.lat.values

        LON, LAT = np.meshgrid(
            lon_values,
            lat_values
        )

        projection = ccrs.PlateCarree()


        # ======================================================
        # PANEL A COLOR SCALE
        # ==============================================================

        days_max = float(
            np.nanpercentile(
                ocean_days,
                99
            )
        )

        days_max = max(
            days_max,
            10
        )

        # Round upward to nearest 10 days
        days_max = (
            np.ceil(
                days_max / 10
            )
            *
            10
        )

        days_levels = np.linspace(
            0,
            days_max,
            15
        )

        days_ticks = np.linspace(
            0,
            days_max,
            7
        )


        # ======================================================
        # PANEL B COLOR SCALE
        #
        # IMPORTANT:
        # Trend must be centered at zero.
        # ==============================================================

        trend_limit = float(
            np.nanpercentile(
                np.abs(
                    ocean_trend
                ),
                99
            )
        )

        trend_limit = max(
            trend_limit,
            1.0
        )

        trend_limit = np.ceil(
            trend_limit
        )

        trend_levels = np.linspace(
            -trend_limit,
            trend_limit,
            15
        )

        trend_ticks = np.linspace(
            -trend_limit,
            trend_limit,
            7
        )


        # ======================================================
        # CREATE FIGURE
        # ==============================================================

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(
                17,
                6.5
            ),
            subplot_kw={
                "projection":
                    projection
            }
        )

        ax1 = axes[0]

        ax2 = axes[1]


        # ======================================================
        # PANEL A: MEAN MHW DAYS
        # ==============================================================

        cf1 = ax1.contourf(
            LON,
            LAT,
            mean_mhw_days,
            levels=days_levels,
            cmap=MAP_CMAP,
            extend="max",
            transform=projection
        )


        # ======================================================
        # PANEL B: MHW DAYS TREND
        # ==============================================================

        cf2 = ax2.contourf(
            LON,
            LAT,
            mhw_days_trend,
            levels=trend_levels,
            cmap=MAP_CMAP,
            extend="both",
            transform=projection
        )


        # ======================================================
        # MAP FORMATTING
        # ==============================================================

        longitude_ticks = [
            -60,
            -50,
            -40,
            -30,
            -20,
            -10
        ]

        latitude_ticks = [
            0,
            5,
            10,
            15,
            20,
            25,
            30
        ]

        for ax in axes:

            ax.set_extent(
                [
                    LON_MIN,
                    LON_MAX,
                    LAT_MIN,
                    LAT_MAX
                ],
                crs=projection
            )

            ax.add_feature(
                cfeature.LAND.with_scale(
                    "50m"
                ),
                facecolor="0.70",
                edgecolor="black",
                linewidth=0.5,
                zorder=10
            )

            ax.coastlines(
                resolution="50m",
                linewidth=0.7,
                zorder=11
            )

            ax.add_feature(
                cfeature.BORDERS.with_scale(
                    "50m"
                ),
                linewidth=0.35,
                edgecolor="black",
                zorder=11
            )

            ax.set_xticks(
                longitude_ticks,
                crs=projection
            )

            ax.set_yticks(
                latitude_ticks,
                crs=projection
            )

            ax.xaxis.set_major_formatter(
                LongitudeFormatter(
                    degree_symbol="°"
                )
            )

            ax.yaxis.set_major_formatter(
                LatitudeFormatter(
                    degree_symbol="°"
                )
            )

            ax.gridlines(
                crs=projection,
                draw_labels=False,
                xlocs=longitude_ticks,
                ylocs=latitude_ticks,
                linewidth=0.3,
                linestyle="--",
                alpha=0.4
            )


        # ======================================================
        # SIGNIFICANCE DOTS
        # ==============================================================

        rows, cols = np.where(
            significant
        )

        # Thin dots to avoid overcrowding
        keep_points = (
            np.arange(
                len(rows)
            )
            %
            5
            ==
            0
        )

        rows = rows[
            keep_points
        ]

        cols = cols[
            keep_points
        ]

        ax2.scatter(
            LON[
                rows,
                cols
            ],
            LAT[
                rows,
                cols
            ],
            s=1.8,
            color="black",
            transform=projection,
            zorder=9
        )


        # ======================================================
        # PANEL TITLES
        # ==============================================================

        ax1.set_title(
            "(a) Mean MHW Days",
            fontsize=13,
            fontweight="bold",
            pad=10
        )

        ax2.set_title(
            "(b) MHW Days Trend",
            fontsize=13,
            fontweight="bold",
            pad=10
        )


        # ======================================================
        # COLORBAR A
        # SAME HEIGHT AS MAP
        # ==============================================================

        cax1 = inset_axes(
            ax1,
            width="3.2%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(
                1.035,
                0,
                1,
                1
            ),
            bbox_transform=ax1.transAxes,
            borderpad=0
        )

        cbar1 = fig.colorbar(
            cf1,
            cax=cax1,
            orientation="vertical",
            ticks=days_ticks
        )

        cbar1.set_label(
            "Days/Year",
            fontsize=10
        )

        cbar1.ax.yaxis.set_major_formatter(
            mticker.FormatStrFormatter(
                "%.1f"
            )
        )

        cbar1.update_ticks()


        # ======================================================
        # COLORBAR B
        # SAME HEIGHT AS MAP
        # ==============================================================

        cax2 = inset_axes(
            ax2,
            width="3.2%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(
                1.035,
                0,
                1,
                1
            ),
            bbox_transform=ax2.transAxes,
            borderpad=0
        )

        cbar2 = fig.colorbar(
            cf2,
            cax=cax2,
            orientation="vertical",
            ticks=trend_ticks
        )

        cbar2.set_label(
            "Days/Decade",
            fontsize=10
        )

        cbar2.ax.yaxis.set_major_formatter(
            mticker.FormatStrFormatter(
                "%.1f"
            )
        )

        cbar2.update_ticks()


        # ======================================================
        # MAIN TITLE
        # ==============================================================

        fig.suptitle(
            "Tropical North East Atlantic Marine Heatwave "
            "Days and Trend (1982–2024)",
            fontsize=16,
            fontweight="bold",
            y=0.97
        )


        # ======================================================
        # SPACING
        # ==============================================================

        fig.subplots_adjust(
            left=0.06,
            right=0.92,
            bottom=0.12,
            top=0.88,
            wspace=0.30
        )

        plt.show()


        # ======================================================
        # TERMINAL RESULTS
        # ==============================================================

        print(
            "\n"
            + "=" * 85
        )

        print(
            "RESULTS"
        )

        print(
            "=" * 85
        )

        print(
            f"Regional mean MHW days: "
            f"{np.nanmean(ocean_days):.2f} days/year"
        )

        print(
            f"Minimum mean MHW days: "
            f"{np.nanmin(ocean_days):.2f} days/year"
        )

        print(
            f"Maximum mean MHW days: "
            f"{np.nanmax(ocean_days):.2f} days/year"
        )

        print(
            f"\nMean MHW days trend: "
            f"{np.nanmean(ocean_trend):+.2f} days/decade"
        )

        print(
            f"Minimum trend: "
            f"{np.nanmin(ocean_trend):+.2f} days/decade"
        )

        print(
            f"Maximum trend: "
            f"{np.nanmax(ocean_trend):+.2f} days/decade"
        )

        significant_percent = (
            np.sum(
                significant
                &
                ocean_mask
            )
            /
            np.sum(
                ocean_mask
            )
            *
            100
        )

        print(
            f"Ocean grid cells with significant trend: "
            f"{significant_percent:.1f}%"
        )

        elapsed = (
            time.perf_counter()
            -
            start_clock
        )

        print(
            f"\nRuntime: "
            f"{elapsed / 60:.1f} minutes"
        )


    except Exception:

        print(
            "\nPROCESS FAILED"
        )

        print(
            traceback.format_exc()
        )


    finally:

        if threshold_ds is not None:
            threshold_ds.close()

        if sst_ds is not None:
            sst_ds.close()


# ==============================================================
# RUN
# ==============================================================

if __name__ == "__main__":

    main()