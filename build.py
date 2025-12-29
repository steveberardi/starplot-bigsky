import logging
import time
from pathlib import Path

import pandas as pd
from shapely.geometry import Point

from starplot import Star
from starplot.data import Catalog, utils


__version__ = "0.1.1"

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data"
BUILD_PATH = HERE / "build"

BIG_SKY_VERSION = "0.4.0"
BIG_SKY_FILENAME = f"bigsky.{BIG_SKY_VERSION}.stars.csv.gz"
BIG_SKY_PQ_FILENAME = f"bigsky.{BIG_SKY_VERSION}.stars.parquet"

BIG_SKY_DOWNLOAD_URL = f"https://github.com/steveberardi/bigsky/releases/download/v{BIG_SKY_VERSION}/{BIG_SKY_FILENAME}"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("build.log", mode="a")
logger.addHandler(console_handler)
logger.addHandler(file_handler)
formatter = logging.Formatter(
    "{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def build_magnitude(limiting_magnitude: float):
    bigsky_path = DATA_PATH / BIG_SKY_FILENAME

    if not bigsky_path.is_file():
        utils.download(
            BIG_SKY_DOWNLOAD_URL,
            bigsky_path,
            "Big Sky Star Catalog",
        )

    output_path = (
        BUILD_PATH / f"stars.bigksy.{__version__}.mag{limiting_magnitude}.parquet"
    )

    df = pd.read_csv(
        bigsky_path,
        header=0,
        usecols=[
            "tyc_id",
            "hip_id",
            "ccdm",
            "magnitude",
            "bv",
            "ra_degrees_j2000",
            "dec_degrees_j2000",
            "ra_mas_per_year",
            "dec_mas_per_year",
            "parallax_mas",
            "constellation",
        ],
        compression="gzip",
    )
    df = df.assign(epoch_year=2000)
    df = df.rename(
        columns={
            "hip_id": "hip",
            "tyc_id": "tyc",
            "ra_degrees_j2000": "ra",
            "dec_degrees_j2000": "dec",
            "constellation": "constellation_id",
        }
    )

    def stars(d):
        ctr = 0
        for star in d.itertuples():
            geometry = Point(star.ra, star.dec)

            if (
                not geometry.is_valid
                or geometry.is_empty
                or star.magnitude > limiting_magnitude
            ):
                continue

            ctr += 1
            yield Star(
                pk=ctr,
                hip=star.hip,
                tyc=star.tyc,
                ra=star.ra,
                dec=star.dec,
                constellation_id=star.constellation_id,
                ccdm=star.ccdm,
                magnitude=star.magnitude,
                parallax_mas=star.parallax_mas,
                ra_mas_per_year=star.ra_mas_per_year or 0,
                dec_mas_per_year=star.dec_mas_per_year or 0,
                bv=star.bv,
                geometry=geometry,
                epoch_year=2000,
            )

        logger.info(f"Magnitude {limiting_magnitude} total = {ctr:,}")

    Catalog.build(
        objects=stars(df),
        path=output_path,
        chunk_size=5_000_000,
        columns=[
            "pk",
            "hip",
            "tyc",
            "ra",
            "dec",
            "magnitude",
            "bv",
            "parallax_mas",
            "ra_mas_per_year",
            "dec_mas_per_year",
            "constellation_id",
            "geometry",
            "ccdm",
            "epoch_year",
        ],
        partition_columns=[],
        sorting_columns=["magnitude", "healpix_index"],
        compression="snappy",
        row_group_size=100_000,
        healpix_nside=4,
    )


def build():
    time_start = time.time()
    logger.info("Building Big Sky - Magnitude 16")
    build_magnitude(16)

    logger.info("Building Big Sky - Magnitude 11")
    build_magnitude(11)

    logger.info("Building Big Sky - Magnitude 9")
    build_magnitude(9)

    duration = time.time() - time_start
    logger.info(f"Done - {duration:.0f}s")


if __name__ == "__main__":
    build()
