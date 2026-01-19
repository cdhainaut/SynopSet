from typing import Union, Optional, Any
from pathlib import Path


class ERA5Downloader:
    """
    A reusable class for downloading ERA5 reanalysis data from the CDS API.
    """

    def __init__(
        self,
        years: list[int],
        months: list[int] = list(range(1, 13)),
        days: list[int] = list(range(1, 32)),
        hours: list[int] = list(range(0, 24)),
        variables: Optional[list[str]] = None,
        lat_bounds: tuple[float, float] = (0.0, 90.0),
        lon_bounds: tuple[float, float] = (-120.0, 30.0),
        dataset: Union[str, Path] = "reanalysis-era5-single-levels",
        output_file: Union[str, Path] = "download.grib",
    ) -> None:
        self.dataset = dataset
        self.output_file = output_file

        self.params: dict[str, Any] = {
            "years": [str(y) for y in years],
            "months": [f"{m:02d}" for m in months],
            "days": [f"{d:02d}" for d in days],
            "hours": [f"{h:02d}:00" for h in hours],
            "variables": variables
            or [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "lat_bounds": lat_bounds,
            "lon_bounds": lon_bounds,
        }

    def _build_area(self) -> list[float]:
        """Returns the area in [N, W, S, E] format required by the CDS API."""
        north = max(self.params["lat_bounds"])
        south = min(self.params["lat_bounds"])
        west = min(self.params["lon_bounds"])
        east = max(self.params["lon_bounds"])
        return [north, west, south, east]

    def _build_request(self) -> dict[str, Any]:
        """Constructs the full CDS API request dictionary."""
        return {
            "product_type": ["reanalysis"],
            "variable": self.params["variables"],
            "year": self.params["years"],
            "month": self.params["months"],
            "day": self.params["days"],
            "time": self.params["hours"],
            "data_format": "grib",
            "download_format": "unarchived",
            "area": self._build_area(),
        }

    def download(self) -> None:
        """Downloads the data using the CDS API."""
        import cdsapi

        print(f"Requesting data from dataset: {self.dataset}")
        client = cdsapi.Client()
        request = self._build_request()
        client.retrieve(self.dataset, request, self.output_file)
        print(f"Download complete: {self.output_file}")
