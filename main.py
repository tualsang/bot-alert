import os
import time
import logging
from datetime import datetime, timedelta, timezone, time as dt_time
from zoneinfo import ZoneInfo

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

# ------------------------------------------------------------
# RSI STRATEGY
# ------------------------------------------------------------

RSI_ENABLED = True

RSI_PERIOD = 14
RSI_HIGH = 75.0
RSI_LOW = 25.0

# RSI is calculated using 1-minute candles.
RSI_TIMEFRAME_MINUTES = 1


# ------------------------------------------------------------
# OPENING RANGE BREAKOUT STRATEGY
# ------------------------------------------------------------

ORB_ENABLED = True

# Opening range:
# 9:30 AM through 9:45 AM Eastern
ORB_START = dt_time(9, 30)
ORB_END = dt_time(9, 45)

# Breakout confirmation candle
ORB_BREAKOUT_TIMEFRAME_MINUTES = 1


# ------------------------------------------------------------
# GENERAL
# ------------------------------------------------------------

# Check Alpaca every 10 seconds.
#
# We still only process COMPLETED candles, so this does not
# generate duplicate signals.
CHECK_INTERVAL_SECONDS = 10

DATA_FEED = DataFeed.IEX

NY_TZ = ZoneInfo("America/New_York")


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

# RSI state
last_processed_rsi_bar = {}
previous_rsi = {}

# ORB state
orb_ranges = {}

# Tracks whether an upside/downside alert has already been
# sent today for each ticker.
orb_alerts = {}

# Last 5-minute candle checked for each ticker.
last_processed_orb_bar = {}

# Current trading date.
current_trading_date = None


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
    try:
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": message},
            timeout=10,
        )

        response.raise_for_status()

        logger.info("Discord alert sent successfully.")

    except requests.RequestException as exc:
        logger.error(
            "Failed to send Discord alert: %s",
            exc,
        )


def send_startup_message():
    message = (
        "🟢 **Market Scanner Started**\n\n"
        f"**Tickers:** {', '.join(TICKERS)}\n\n"

        "📊 **Strategy 1 — RSI**\n"
        f"Timeframe: {RSI_TIMEFRAME_MINUTES}m\n"
        f"RSI Period: {RSI_PERIOD}\n"
        f"Overbought: {RSI_HIGH:g}\n"
        f"Oversold: {RSI_LOW:g}\n\n"

        "📈 **Strategy 2 — Opening Range Breakout**\n"
        "Opening Range: 9:30–9:45 AM ET\n"
        f"Confirmation: {ORB_BREAKOUT_TIMEFRAME_MINUTES}m candle close\n"
        "Alert above OR high or below OR low\n\n"
    )

    send_discord_message(message)


# ============================================================
# RSI ALERTS
# ============================================================

def send_rsi_high_alert(
    ticker,
    price,
    rsi,
    previous,
    timestamp,
):
    message = (
        "🔴 ** {{ticker}} OVERBOUGHT**\n\n"
        f"RSI({RSI_PERIOD}): **{rsi:.2f}**\n"
        f"Previous RSI: {previous:.2f}\n"
        f"Timeframe: **{RSI_TIMEFRAME_MINUTES}m**\n"
        f"Threshold: **>{RSI_HIGH:g}**\n\n"
    )

    send_discord_message(message)


def send_rsi_low_alert(
    ticker,
    price,
    rsi,
    previous,
    timestamp,
):
    message = (
        "🟢 **{{ticker}} OVERSOLD**\n\n"
        f"RSI({RSI_PERIOD}): **{rsi:.2f}**\n"
        f"Previous RSI: {previous:.2f}\n"
        f"Timeframe: **{RSI_TIMEFRAME_MINUTES}m**\n"
        f"Threshold: **<{RSI_LOW:g}**\n\n"
    )

    send_discord_message(message)


# ============================================================
# ORB ALERTS
# ============================================================

def send_orb_high_alert(
    ticker,
    close_price,
    range_high,
    range_low,
    timestamp,
):
    message = (
        "🚀 ** {{ticker}}  RANGE BREAKOUT**\n\n"
        f"1m Close: **${close_price:,.2f}**\n\n"
        f"Opening Range High: **${range_high:,.2f}**\n"
        f"Opening Range Low: ${range_low:,.2f}\n\n"
    )

    send_discord_message(message)


def send_orb_low_alert(
    ticker,
    close_price,
    range_high,
    range_low,
    timestamp,
):
    message = (
        "📉 ** {{ticker}} OPENING RANGE BREAKDOWN**\n\n"
        f"1m Close: **${close_price:,.2f}**\n\n"
        f"Opening Range High: ${range_high:,.2f}\n"
        f"Opening Range Low: **${range_low:,.2f}**\n\n"
    )

    send_discord_message(message)

# ============================================================
# ALPACA DATA
# ============================================================

def get_all_bars():
    """
    ONE Alpaca request.

    Get 1-minute bars for every symbol.

    We use these bars for:
      1. 1-minute RSI
      2. Calculating the 9:30-9:45 opening range
      3. Building local 5-minute candles for ORB
    """

    end = datetime.now(timezone.utc)

    # Two calendar days gives plenty of 1-minute data for RSI,
    # weekends notwithstanding.
    start = end - timedelta(days=2)

    request = StockBarsRequest(
        symbol_or_symbols=TICKERS,
        timeframe=TimeFrame(
            1,
            TimeFrameUnit.Minute,
        ),
        start=start,
        end=end,
        feed=DATA_FEED,
    )

    bars = alpaca_client.get_stock_bars(request)

    return bars.df


# ============================================================
# DATA HELPERS
# ============================================================

def get_symbol_dataframe(
    all_bars: pd.DataFrame,
    ticker: str,
):
    """
    Extract one ticker from Alpaca's multi-symbol dataframe.
    """

    if all_bars.empty:
        return pd.DataFrame()

    if not isinstance(
        all_bars.index,
        pd.MultiIndex,
    ):
        return pd.DataFrame()

    try:
        df = all_bars.xs(
            ticker,
            level="symbol",
        ).copy()

    except KeyError:
        return pd.DataFrame()

    df.index = pd.to_datetime(
        df.index,
        utc=True,
    )

    df = df.sort_index()

    # Convert timestamps to Eastern Time.
    df.index = df.index.tz_convert(
        NY_TZ
    )

    return df


def get_today_regular_session(df):
    """
    Keep only today's regular-session data beginning at 9:30 AM.
    """

    if df.empty:
        return df

    today = datetime.now(
        NY_TZ
    ).date()

    df = df[
        df.index.date == today
    ]

    df = df[
        df.index.time >= dt_time(9, 30)
    ]

    df = df[
        df.index.time < dt_time(16, 0)
    ]

    return df


# ============================================================
# RSI STRATEGY
# ============================================================

def process_rsi(
    ticker,
    df,
):
    if not RSI_ENABLED:
        return

    if len(df) < RSI_PERIOD + 2:
        return

    # Remove current incomplete 1-minute candle.
    now = pd.Timestamp.now(
        tz=NY_TZ
    )

    current_minute = now.floor("1min")

    completed_df = df[
        df.index < current_minute
    ].copy()

    if len(completed_df) < RSI_PERIOD + 2:
        return

    completed_df["rsi"] = calculate_rsi(
        completed_df["close"],
        RSI_PERIOD,
    )

    completed_df = completed_df.dropna(
        subset=["rsi"]
    )

    if len(completed_df) < 2:
        return

    latest = completed_df.iloc[-1]
    latest_timestamp = completed_df.index[-1]

    if (
        last_processed_rsi_bar.get(ticker)
        == latest_timestamp
    ):
        return

    last_processed_rsi_bar[
        ticker
    ] = latest_timestamp

    current_rsi = float(
        latest["rsi"]
    )

    price = float(
        latest["close"]
    )

    old_rsi = previous_rsi.get(
        ticker
    )

    if old_rsi is None:
        old_rsi = float(
            completed_df.iloc[-2]["rsi"]
        )

    logger.info(
        "%s | 1m Price %.2f | RSI %.2f",
        ticker,
        price,
        current_rsi,
    )

    if (
        old_rsi < RSI_HIGH
        and current_rsi >= RSI_HIGH
    ):
        send_rsi_high_alert(
            ticker,
            price,
            current_rsi,
            old_rsi,
            latest_timestamp,
        )

    elif (
        old_rsi > RSI_LOW
        and current_rsi <= RSI_LOW
    ):
        send_rsi_low_alert(
            ticker,
            price,
            current_rsi,
            old_rsi,
            latest_timestamp,
        )

    previous_rsi[
        ticker
    ] = current_rsi


# ============================================================
# OPENING RANGE
# ============================================================

def calculate_opening_range(
    ticker,
    df,
):
    """
    Calculate the high and low from 9:30 through 9:44.

    The 9:30 15-minute candle completes at 9:45.
    """

    if not ORB_ENABLED:
        return None

    now = datetime.now(
        NY_TZ
    )

    if now.time() < ORB_END:
        return None

    range_df = df[
        (df.index.time >= ORB_START)
        & (df.index.time < ORB_END)
    ]

    if range_df.empty:
        return None

    range_high = float(
        range_df["high"].max()
    )

    range_low = float(
        range_df["low"].min()
    )

    today = now.date()

    existing = orb_ranges.get(
        ticker
    )

    # Only set the range once per day.
    if (
        existing is None
        or existing["date"] != today
    ):
        orb_ranges[ticker] = {
            "date": today,
            "high": range_high,
            "low": range_low,
        }

        logger.info(
            "%s | Opening Range locked | High %.2f | Low %.2f",
            ticker,
            range_high,
            range_low,
        )

    return orb_ranges[ticker]


# ============================================================
# ORB STRATEGY
# ============================================================

def process_orb(
    ticker,
    df,
):
    if not ORB_ENABLED:
        return

    now = datetime.now(
        NY_TZ
    )

    # Opening range completes at 9:45.
    # The first eligible breakout candle is 9:45-9:46.
    if now.time() < dt_time(9, 46):
        return

    opening_range = calculate_opening_range(
        ticker,
        df,
    )

    if not opening_range:
        return

    range_high = opening_range["high"]
    range_low = opening_range["low"]

    # Remove the currently-forming 1-minute candle.
    current_minute = pd.Timestamp.now(
        tz=NY_TZ
    ).floor("1min")

    completed_bars = df[
        df.index < current_minute
    ].copy()

    # Only check candles that begin at 9:45 or later.
    completed_bars = completed_bars[
        completed_bars.index.time >= dt_time(9, 45)
    ]

    if completed_bars.empty:
        return

    latest = completed_bars.iloc[-1]
    latest_timestamp = completed_bars.index[-1]

    # Don't process the same 1-minute candle twice.
    if (
        last_processed_orb_bar.get(ticker)
        == latest_timestamp
    ):
        return

    last_processed_orb_bar[ticker] = latest_timestamp

    close_price = float(
        latest["close"]
    )

    logger.info(
        "%s | ORB | 1m Close %.2f | Range %.2f - %.2f",
        ticker,
        close_price,
        range_low,
        range_high,
    )

    today = now.date()

    if (
        ticker not in orb_alerts
        or orb_alerts[ticker]["date"] != today
    ):
        orb_alerts[ticker] = {
            "date": today,
            "above": False,
            "below": False,
        }

    # UPSIDE BREAKOUT
    if (
        close_price > range_high
        and not orb_alerts[ticker]["above"]
    ):
        orb_alerts[ticker]["above"] = True

        send_orb_high_alert(
            ticker,
            close_price,
            range_high,
            range_low,
            latest_timestamp,
        )

    # DOWNSIDE BREAKDOWN
    elif (
        close_price < range_low
        and not orb_alerts[ticker]["below"]
    ):
        orb_alerts[ticker]["below"] = True

        send_orb_low_alert(
            ticker,
            close_price,
            range_high,
            range_low,
            latest_timestamp,
        )


# ============================================================
# RESET EACH NEW TRADING DAY
# ============================================================

def reset_daily_state_if_needed():
    global current_trading_date

    today = datetime.now(
        NY_TZ
    ).date()

    if current_trading_date == today:
        return

    current_trading_date = today

    orb_ranges.clear()
    orb_alerts.clear()
    last_processed_orb_bar.clear()

    logger.info(
        "New trading day detected. ORB state reset."
    )


# ============================================================
# MAIN
# ============================================================

def run():
    logger.info(
        "Starting combined market scanner..."
    )

    logger.info(
        "Watching: %s",
        ", ".join(TICKERS),
    )

    logger.info(
        "RSI: 1m RSI(%d), thresholds %.1f / %.1f",
        RSI_PERIOD,
        RSI_LOW,
        RSI_HIGH,
    )

    logger.info(
        "ORB: 9:30-9:45 range, 5m close confirmation"
    )

    send_startup_message()

    while True:
        try:
            reset_daily_state_if_needed()

            now = datetime.now(
                NY_TZ
            )

            # Only scan around regular market hours.
            #
            # Starts a little before market open so the bot can
            # already be running when 9:30 arrives.
            if (
                now.weekday() >= 5
                or now.time() < dt_time(9, 25)
                or now.time() >= dt_time(16, 5)
            ):
                logger.info(
                    "Outside market hours. Sleeping 60 seconds."
                )

                time.sleep(60)
                continue

            logger.info(
                "Fetching 1m data for all 9 symbols..."
            )

            # ONE API CALL
            all_bars = get_all_bars()

            for ticker in TICKERS:
                df = get_symbol_dataframe(
                    all_bars,
                    ticker,
                )

                if df.empty:
                    logger.warning(
                        "%s: No data.",
                        ticker,
                    )
                    continue

                regular_df = get_today_regular_session(
                    df
                )

                # RSI needs previous historical bars too,
                # so use the full dataframe.
                process_rsi(
                    ticker,
                    df,
                )

                # ORB only uses today's regular session.
                process_orb(
                    ticker,
                    regular_df,
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
                "Unexpected error in scanner."
            )

            time.sleep(30)


if __name__ == "__main__":
    run()