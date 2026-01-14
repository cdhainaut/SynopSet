from nomad.environment import GRIB
from nomad.tools.era5_downloader import ERA5Downloader
from pathlib import Path
import numpy as np

path_execution = Path(__file__).parent

wind_file = "wind_2020.grib"
wave_file = "wave_2020.grib"

# hours = list(np.arange(0, 24, 12))
hours = [0, 6, 12, 18]
lon_bounds = (-80, 0)
lat_bounds = (20, 60)

wind_downloader = ERA5Downloader(
    output_file=wind_file,
    variables=[
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ],
    years=[2020],
    hours=hours,
    lat_bounds=lat_bounds,
    lon_bounds=lon_bounds,
)
wind_downloader.download()

wave_downloader = ERA5Downloader(
    output_file=wave_file,
    variables=[
        "mean_wave_direction",
        "mean_wave_period",
        "significant_height_of_combined_wind_waves_and_swell",
    ],
    years=[2020],
    hours=hours,
    lat_bounds=lat_bounds,
    lon_bounds=lon_bounds,
)
wave_downloader.download()
