from datetime import datetime
from io import StringIO

import pandas as pd

from .alpha_vantage_common import _filter_csv_by_date_range, _make_api_request
from .errors import NoMarketDataError


def get_stock(
    symbol: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Returns raw daily OHLCV values, adjusted close values, and historical split/dividend events
    filtered to the specified date range.

    Args:
        symbol: The name of the equity. For example: symbol=IBM
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        CSV string containing the daily adjusted time series data filtered to the date range.
    """
    # Parse dates to determine the range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    today = datetime.now()

    # Choose outputsize based on whether the requested range is within the latest 100 days
    # Compact returns latest 100 data points, so check if start_date is recent enough
    days_from_today_to_start = (today - start_dt).days
    outputsize = "compact" if days_from_today_to_start < 100 else "full"

    params = {
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": "csv",
    }

    response = _make_api_request("TIME_SERIES_DAILY_ADJUSTED", params)

    return _filter_csv_by_date_range(response, start_date, end_date)


def get_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Normalize Alpha Vantage daily adjusted CSV to the shared price basis."""
    response = _make_api_request("TIME_SERIES_DAILY_ADJUSTED", {
        "symbol": symbol, "outputsize": "full", "datatype": "csv",
    })
    data = pd.read_csv(StringIO(response))
    required = {"timestamp", "open", "high", "low", "close", "adjusted_close", "volume"}
    if data.empty or not required.issubset(data.columns):
        raise NoMarketDataError(symbol, detail="Alpha Vantage returned no adjusted daily bars")
    factor = pd.to_numeric(data["adjusted_close"], errors="coerce") / pd.to_numeric(data["close"], errors="coerce")
    if factor.isna().any() or (factor <= 0).any():
        raise NoMarketDataError(symbol, detail="Alpha Vantage adjustment factors are unavailable")
    frame = pd.DataFrame({"Date": data["timestamp"], "Volume": data["volume"]})
    for name in ("open", "high", "low", "close"):
        frame[name.title()] = pd.to_numeric(data[name], errors="coerce") * factor
    return frame
