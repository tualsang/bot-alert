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

TIMEFRAME_MINUTES = 1
CHECK_INTERVAL_SECONDS = 10

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
# ENVIRONMENT VARIABLES
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


validate_environment()


# ============================================================
# ALPACA CLIENT
# ============================================================

alpaca_client = StockHistoricalDataClient(
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
)


# ============================================================
# STATE
# ============================================================

last_processed_bar = {}
previous_rsi = {}


# ============================================================
# RSI
# ============================================================

def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
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
    try:
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": message},
            timeout=10,
        )

        response.raise_for_status()

        logger.info("Discord alert sent successfully.")

    except requests.RequestException as exc:
        logger.error("Failed to send Discord alert: %s", exc)


def send_startup_message():
    message = (
        "🟢 **RSI Scanner Started**\n\n"
        f"**Tickers:** {', '.join(TICKERS)}\n"
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
        "🔴 **{{ticker}} OVERBOUGHT **\n\n"
        f"RSI({RSI_PERIOD}): **{rsi:.2f}**\n"
        f"Previous RSI: {previous:.2f}\n"
        f"Timeframe: **{TIMEFRAME_MINUTES}m**\n"
        f"Threshold: **>{RSI_HIGH:g}**\n"
        f"Candle: `{timestamp}`\n\n"
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
        "🟢 **{{ticker}} RSI OVERSOLD**\n\n"
        f"RSI({RSI_PERIOD}): **{rsi:.2f}**\n"
        f"Previous RSI: {previous:.2f}\n"
        f"Timeframe: **{TIMEFRAME_MINUTES}m**\n"
        f"Threshold: **<{RSI_LOW:g}**\n"
        f"Candle: `{timestamp}`\n\n"
    )

    send_discord_message(message)


# ============================================================
# ONE ALPACA REQUEST FOR ALL SYMBOLS
# ============================================================

def get_all_bars():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=3)

    request = StockBarsRequest(
        symbol_or_symbols=TICKERS,
        timeframe=TimeFrame(
            TIMEFRAME_MINUTES,
            TimeFrameUnit.Minute,
        ),
        start=start,
        end=end,
        feed=DATA_FEED,
    )

    bars = alpaca_client.get_stock_bars(request)

    return bars.df


# ============================================================
# REMOVE INCOMPLETE CURRENT CANDLE
# ============================================================

def remove_current_incomplete_bar(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    df.index = pd.to_datetime(
        df.index,
        utc=True,
    )

    now = pd.Timestamp.now(tz="UTC")

    current_bucket = now.floor(
        f"{TIMEFRAME_MINUTES}min"
    )

    return df[df.index < current_bucket]


# ============================================================
# PROCESS ONE SYMBOL
# ============================================================

def process_ticker(ticker: str, all_bars: pd.DataFrame):
    try:
        if all_bars.empty:
            logger.warning("No market data returned.")
            return

        if not isinstance(all_bars.index, pd.MultiIndex):
            logger.warning(
                "Unexpected Alpaca response format."
            )
            return

        try:
            df = all_bars.xs(
                ticker,
                level="symbol",
            ).copy()

        except KeyError:
            logger.warning(
                "%s: No data returned.",
                ticker,
            )
            return

        df = df.sort_index()

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

        df = df.dropna(
            subset=["rsi"]
        )

        if len(df) < 2:
            return

        latest = df.iloc[-1]

        latest_timestamp = df.index[-1]

        current_rsi = float(
            latest["rsi"]
        )

        current_price = float(
            latest["close"]
        )

        # Skip if we already processed this exact candle.
        if (
            last_processed_bar.get(ticker)
            == latest_timestamp
        ):
            return

        last_processed_bar[
            ticker
        ] = latest_timestamp

        logger.info(
            "%s | Price %.2f | RSI %.2f | Candle %s",
            ticker,
            current_price,
            current_rsi,
            latest_timestamp,
        )

        old_rsi = previous_rsi.get(
            ticker
        )

        # Establish baseline on startup.
        if old_rsi is None:
            old_rsi = float(
                df.iloc[-2]["rsi"]
            )

        # Cross ABOVE 75.
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

        # Cross BELOW 25.
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

        previous_rsi[
            ticker
        ] = current_rsi

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
            logger.info(
                "Fetching all symbols from Alpaca..."
            )

            all_bars = get_all_bars()

            for ticker in TICKERS:
                process_ticker(
                    ticker,
                    all_bars,
                )

            logger.info(
                "Scan complete. Sleeping %d seconds.",
                CHECK_INTERVAL_SECONDS,
            )

            time.sleep(
                CHECK_INTERVAL_SECONDS
            )

        except KeyboardInterrupt:
            logger.info(
                "Scanner stopped by user."
            )
            break

        except Exception:
            logger.exception(
                "Unexpected error in main scanner loop."
            )

            time.sleep(30)


if __name__ == "__main__":
    run()