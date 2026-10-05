import os
import time
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed


# ============================================================
# CONFIGURATION
# ============================================================

TICKERS = [
    "NVDA",
    "AAPL",
    "META",
    "GOOGL",
    "AMZN",
    "MSFT",
    "TSLA",
    "SPY",
    "QQQ",
]

RSI_PERIOD = 14

RSI_HIGH = 75.0
RSI_LOW = 25.0

TIMEFRAME_MINUTES = 5

# Number of seconds between checks.
# We check often, but only process a ticker when a new completed
# 5-minute candle becomes available.
CHECK_INTERVAL_SECONDS = 30

# Alpaca free market-data users can use IEX.
DATA_FEED = DataFeed.IEX


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def validate_environment():
    missing = []

    if not ALPACA_API_KEY:
        missing.append("ALPACA_API_KEY")

    if not ALPACA_API_SECRET:
        missing.append("ALPACA_API_SECRET")

    if not DISCORD_WEBHOOK_URL:
        missing.append("DISCORD_WEBHOOK_URL")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )


# ============================================================
# ALPACA CLIENT
# ============================================================

validate_environment()

alpaca_client = StockHistoricalDataClient(
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
)


# ============================================================
# STATE
# ============================================================

# Last completed candle timestamp processed for each ticker.
last_processed_bar = {}

# Previous RSI value for each ticker.
previous_rsi = {}


# ============================================================
# RSI CALCULATION
# ============================================================

def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI using Wilder-style exponential smoothing.
    """

    delta = close_prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = average_gain / average_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# DISCORD
# ============================================================

def send_discord_message(message: str):
    payload = {
        "content": message
    }

    try:
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json=payload,
            timeout=10,
        )

        response.raise_for_status()

        logger.info("Discord alert sent successfully.")

    except requests.RequestException as exc:
        logger.error("Failed to send Discord alert: %s", exc)


def send_startup_message():
    tickers = ", ".join(TICKERS)

    message = (
        "🟢 **RSI Scanner Started**\n\n"
        f"**Tickers:** {tickers}\n"
        f"**Timeframe:** {TIMEFRAME_MINUTES}m\n"
        f"**RSI Period:** {RSI_PERIOD}\n"
        f"**Overbought:** {RSI_HIGH:g}\n"
        f"**Oversold:** {RSI_LOW:g}\n"
        "**Feed:** Alpaca IEX\n\n"
        "I'll notify you when RSI crosses one of the thresholds."
    )

    send_discord_message(message)


def send_high_alert(
    ticker: str,
    price: float,
    rsi: float,
    previous: float,
    timestamp,
):
    message = (
        "🔴 **RSI OVERBOUGHT ALERT**\n\n"
        f"**{ticker}**\n"
        f"Price: **${price:,.2f}**\n"
        f"RSI({RSI_PERIOD}): **{rsi:.2f}**\n"
        f"Previous RSI: {previous:.2f}\n"
        f"Timeframe: **{TIMEFRAME_MINUTES}m**\n"
        f"Threshold: **>{RSI_HIGH:g}**\n"
        f"Candle: `{timestamp}`\n\n"
        f"{ticker} crossed ABOVE RSI {RSI_HIGH:g}."
    )

    send_discord_message(message)


def send_low_alert(
    ticker: str,
    price: float,
    rsi: float,
    previous: float,
    timestamp,
):
    message = (
        "🟢 **RSI OVERSOLD ALERT**\n\n"
        f"**{ticker}**\n"
        f"Price: **${price:,.2f}**\n"
        f"RSI({RSI_PERIOD}): **{rsi:.2f}**\n"
        f"Previous RSI: {previous:.2f}\n"
        f"Timeframe: **{TIMEFRAME_MINUTES}m**\n"
        f"Threshold: **<{RSI_LOW:g}**\n"
        f"Candle: `{timestamp}`\n\n"
        f"{ticker} crossed BELOW RSI {RSI_LOW:g}."
    )

    send_discord_message(message)


# ============================================================
# MARKET DATA
# ============================================================

def get_bars(ticker: str) -> pd.DataFrame:
    """
    Retrieve enough 5-minute bars to calculate RSI reliably.
    """

    end = datetime.now(timezone.utc)

    # Several calendar days gives us enough regular-market
    # 5-minute candles even over weekends.
    start = end - timedelta(days=7)

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(
            TIMEFRAME_MINUTES,
            TimeFrameUnit.Minute,
        ),
        start=start,
        end=end,
        feed=DATA_FEED,
    )

    bars = alpaca_client.get_stock_bars(request)

    df = bars.df

    if df.empty:
        return pd.DataFrame()

    # Alpaca may return a MultiIndex:
    # symbol + timestamp
    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.xs(ticker)
        except KeyError:
            return pd.DataFrame()

    return df.sort_index()


# ============================================================
# CANDLE HANDLING
# ============================================================

def remove_current_incomplete_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Avoid calculating RSI from the currently forming candle.

    Example:
    If the current time is 10:07, the 10:05-10:10 candle
    is still forming, so we discard it.
    """

    if df.empty:
        return df

    now = pd.Timestamp.now(tz="UTC")

    current_bucket = now.floor(
        f"{TIMEFRAME_MINUTES}min"
    )

    df = df.copy()

    # Convert index to UTC timestamps if necessary.
    index = pd.to_datetime(df.index, utc=True)

    df.index = index

    # Bars starting at the current bucket are incomplete.
    df = df[df.index < current_bucket]

    return df


# ============================================================
# PROCESS TICKER
# ============================================================

def process_ticker(ticker: str):
    try:
        df = get_bars(ticker)

        if df.empty:
            logger.warning("%s: No market data returned.", ticker)
            return

        df = remove_current_incomplete_bar(df)

        if len(df) < RSI_PERIOD + 2:
            logger.warning(
                "%s: Not enough candles for RSI. Got %d.",
                ticker,
                len(df),
            )
            return

        df["rsi"] = calculate_rsi(
            df["close"],
            RSI_PERIOD,
        )

        df = df.dropna(subset=["rsi"])

        if len(df) < 2:
            return

        latest = df.iloc[-1]

        latest_timestamp = df.index[-1]

        current_rsi = float(latest["rsi"])
        current_price = float(latest["close"])

        # Don't process the same completed candle twice.
        if last_processed_bar.get(ticker) == latest_timestamp:
            return

        last_processed_bar[ticker] = latest_timestamp

        logger.info(
            "%s | Price %.2f | RSI %.2f | Candle %s",
            ticker,
            current_price,
            current_rsi,
            latest_timestamp,
        )

        old_rsi = previous_rsi.get(ticker)

        # On startup we need a baseline.
        # We intentionally do not send an alert just because
        # RSI is already above/below a threshold.
        if old_rsi is None:
            if len(df) >= 2:
                old_rsi = float(df.iloc[-2]["rsi"])
            else:
                previous_rsi[ticker] = current_rsi
                return

        # Crossing ABOVE 75.
        if (
            old_rsi < RSI_HIGH
            and current_rsi >= RSI_HIGH
        ):
            send_high_alert(
                ticker=ticker,
                price=current_price,
                rsi=current_rsi,
                previous=old_rsi,
                timestamp=latest_timestamp,
            )

        # Crossing BELOW 25.
        elif (
            old_rsi > RSI_LOW
            and current_rsi <= RSI_LOW
        ):
            send_low_alert(
                ticker=ticker,
                price=current_price,
                rsi=current_rsi,
                previous=old_rsi,
                timestamp=latest_timestamp,
            )

        previous_rsi[ticker] = current_rsi

    except Exception:
        logger.exception(
            "Unexpected error processing %s.",
            ticker,
        )


# ============================================================
# MAIN LOOP
# ============================================================

def run():
    logger.info("Starting RSI scanner...")

    logger.info(
        "Watching %d symbols: %s",
        len(TICKERS),
        ", ".join(TICKERS),
    )

    logger.info(
        "RSI(%d), %dm timeframe, thresholds %.1f / %.1f",
        RSI_PERIOD,
        TIMEFRAME_MINUTES,
        RSI_LOW,
        RSI_HIGH,
    )

    send_startup_message()

    while True:
        try:
            for ticker in TICKERS:
                process_ticker(ticker)

            logger.info(
                "Scan complete. Sleeping %d seconds.",
                CHECK_INTERVAL_SECONDS,
            )

            time.sleep(CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Scanner stopped by user.")
            break

        except Exception:
            logger.exception(
                "Unexpected error in main scanner loop."
            )

            time.sleep(30)


if __name__ == "__main__":
    run()