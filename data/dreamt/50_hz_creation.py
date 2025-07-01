import os
from pathlib import Path
import polars as pl
import numpy as np
from tqdm import tqdm

import pisces.data_sets as pds

timestamp_col = "TIMESTAMP"
psg_col = "Sleep_Stage"
x_col = "ACC_X"
y_col = "ACC_Y"
z_col = "ACC_Z"

SAVE_HZ = 64
BASE_HZ = 64
ACC_G_FACTOR = 1/64
"""
From Metadata.txt, we have these meanings for the classes. We want to convert these to numeric scores by setting all the sleeps to class 0, the "W" to class 1, and the rest to -1

Class meanings:
	    acronym: "P",
            full_form: "Preparation",

	    acronym: "W",
            full_form: "Wake",

	    acronym: "N1",
            full_form: "non-rapid eye-movement (NREM) stage 1",

	    acronym: "N2",
            full_form: "non-rapid eye-movement (NREM) stage 2",

	    acronym: "N3",
            full_form: "non-rapid eye-movement (NREM) stage 3",

	    acronym: "R",
            full_form: "Rapid eye movement (REM)",

	    acronym: "Missing",
            full_form: "No sleep stage labeled",
"""

def map_psg_to_numeric(psg_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Map the PSG data to numeric values.
    """
    psg_raw = psg_raw.with_columns(
        pl.when(pl.col(psg_col) == "Missing").then(-1)
        .when(pl.col(psg_col).is_in(["P", "W"])).then(1)
        .when(pl.col(psg_col).is_in(["N1", "N2", "N3", "R"])).then(0)
        .otherwise(-1).alias(psg_col)
    )
    return psg_raw

def resample_psg(psg_raw: pl.DataFrame, every_ms: int) -> pl.DataFrame:
    """
    Resample the PSG data to 50Hz.
    """
    # timestamp_col is measured in seconds after start, 0.0+
    # change to a time series starting at unix time 0

    psg_dttm = psg_raw.with_columns(
        # convert the timestamp to datetime
        (pl.col(timestamp_col
               ) * 1e3).cast(
                   pl.Datetime(
                       time_unit='ms',
                       time_zone="America/New_York"
                       )).alias(timestamp_col),
        # convert the psg_col to a numerical mask=-1/sleep=0/wake=1
    ).sort(timestamp_col)
    psg_dttm = map_psg_to_numeric(psg_dttm)

    psg_resampled = (
       psg_dttm 
        .group_by_dynamic(
            timestamp_col,
            every=f"{every_ms}ms",
            closed="right",
            include_boundaries=True,
        )
        .agg(pl.col(psg_col).mean().round(0).alias(psg_col))
    )
    return psg_resampled.select(pl.col(timestamp_col).dt.epoch(time_unit='s'), psg_col)

def resample_acc(acc_raw: pl.DataFrame, every_ms: int) -> pl.DataFrame:
    """
    Resample the accelerometer data to 50Hz.
    """
    # timestamp_col is measured in seconds after start, 0.0+
    # change to a time series starting at unix time 0

    acc_dttm = acc_raw.with_columns(
        # convert the timestamp to datetime
        (pl.col(timestamp_col
               ) * 1e3).cast(
                   pl.Datetime(
                       time_unit='ms',
                       time_zone="America/New_York"
                       )).alias(timestamp_col),
    ).sort(timestamp_col)

    acc_resampled = (
       acc_dttm 
        .group_by_dynamic(
            timestamp_col,
            every=f"{every_ms}ms",
            closed="right",
            include_boundaries=True,
        )
        .agg(pl.col(x_col).mean().round(0).alias(x_col),
             pl.col(y_col).mean().round(0).alias(y_col),
             pl.col(z_col).mean().round(0).alias(z_col))
    )
    return acc_resampled.select(
        pl.col(timestamp_col).dt.epoch(time_unit='ms') / 1000,
        pl.col(x_col) * ACC_G_FACTOR,
        pl.col(y_col) * ACC_G_FACTOR,
        pl.col(z_col) * ACC_G_FACTOR)

# Process CSV files from cleaned_dfs directory
cleaned_dfs_path = Path("cleaned_dfs")
cleaned_accelerometer_path = Path("cleaned_accelerometer")
cleaned_psg_path = Path("cleaned_psg")
cleaned_hr_path = Path("cleaned_hr")

if not cleaned_accelerometer_path.exists():
    os.makedirs(cleaned_accelerometer_path, exist_ok=True)
    os.makedirs(cleaned_psg_path, exist_ok=True)

    # Add logic to extract HR data
    os.makedirs(cleaned_hr_path, exist_ok=True)

    # Update progress bars for better clarity
    for csv_file in tqdm(cleaned_dfs_path.glob("*.csv"), desc="Processing cleaned_dfs", unit="file"):
        # Read the CSV file
        df = pl.read_csv(csv_file)

        # Extract accelerometer data
        acc_df = df.select(["TIMESTAMP", "ACC_X", "ACC_Y", "ACC_Z"])
        acc_output_path = cleaned_accelerometer_path / csv_file.name
        acc_df.write_csv(acc_output_path)

        # Extract PSG data
        psg_df = df.select(["TIMESTAMP", "Sleep_Stage"])
        psg_output_path = cleaned_psg_path / csv_file.name
        psg_df.write_csv(psg_output_path)

        # Extract HR data
        hr_df = df.select(["TIMESTAMP", "HR"])
        hr_output_path = cleaned_hr_path / csv_file.name
        hr_df.write_csv(hr_output_path)

# load each csv file inside ./cleaned_accelerometer and ./cleaned_psg
# resample to N hz
# write into ../dreamt_Nhz/cleaned_*

sets = pds.DataSetObject.find_data_sets("/home/eric/Engineering/Work/pisces/data", try_parse=False)
dreamt_data = sets['dreamt']
dreamt_data.parse_data(id_templates="<<ID>>_whole_df.csv")

psg_path = Path(f"../dreamt_{SAVE_HZ}hz/cleaned_psg")
acc_path = Path(f"../dreamt_{SAVE_HZ}hz/cleaned_accelerometer")
hr_path = Path(f"../dreamt_{SAVE_HZ}hz/cleaned_hr")

os.makedirs(psg_path, exist_ok=True)
os.makedirs(acc_path, exist_ok=True)
os.makedirs(hr_path, exist_ok=True)

for d_id in tqdm(dreamt_data.ids, desc="Processing dreamt data", unit="dataset"):
    try:
        d_id_psg = dreamt_data.get_feature_data("psg", d_id)

        psg_s = 30
        psg_ms = psg_s * 1000

        d_id_binary = map_psg_to_numeric(d_id_psg)
        d_id_psg_resampled = resample_psg(d_id_psg, every_ms=psg_ms)

        d_id_psg_resampled\
            .filter(pl.col(timestamp_col) >= 0)\
            .write_csv(
                psg_path / f"{d_id}.csv")

        d_id_acc = dreamt_data.get_feature_data("accelerometer", d_id)
        if SAVE_HZ != BASE_HZ:
            # resample
            d_id_acc = resample_acc(d_id_acc, every_ms=int(1000 * 1 / SAVE_HZ))

        d_id_acc\
            .filter(pl.col(timestamp_col) >= 0)\
            .write_csv(
            acc_path / f"{d_id}.csv")

        # now do heart rate
        d_id_hr = dreamt_data.get_feature_data("hr", d_id)
        d_id_hr = d_id_hr.with_columns(
            (pl.col(timestamp_col) * 1e3).cast(
                pl.Datetime(
                    time_unit='ms',
                    time_zone="America/New_York"
                    )).alias(timestamp_col),
        )      
        d_id_hr_resampled = (
            d_id_hr
            .group_by_dynamic(
                timestamp_col,
                every=f"{int(1000 * 1 / SAVE_HZ)}ms",
                closed="right",
                include_boundaries=True,
            )
            .agg(pl.col("HR").max().round(1).alias("HR"))
        )

        d_id_hr_resampled = d_id_hr_resampled.select(
            pl.col(timestamp_col).dt.epoch(time_unit='ms') / 1000,
            pl.col("HR")
        )
        d_id_hr_resampled\
            .filter(pl.col(timestamp_col) >= 0)\
            .write_csv(
                hr_path/ f"{d_id}.csv")
    except Exception as e:
        print(f"Error processing {d_id}: {e}")
        continue
