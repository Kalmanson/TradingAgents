import logging
from typing import Annotated

from langchain_core.tools import tool
from requests import RequestException

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.interface import route_to_vendor

logger = logging.getLogger(__name__)
NEWS_UNAVAILABLE_PREFIX = "NEWS_DATA_UNAVAILABLE:"


def _get_optional_news(method: str, *args):
    """Keep configured vendor fallback, but let analysts proceed without news."""
    try:
        return route_to_vendor(method, *args)
    except (VendorError, RequestException) as exc:
        # Do not expose provider messages, which may contain credentials or URLs.
        logger.warning(
            "News source unavailable event=news_data_unavailable tool=%s error_type=%s",
            method, type(exc).__name__,
        )
        return (
            f"{NEWS_UNAVAILABLE_PREFIX} {method} failed ({type(exc).__name__}). "
            "News coverage is unavailable from the configured sources. This does not "
            "mean there were no news events or that sentiment is neutral. Continue "
            "with other available evidence; explicitly disclose this coverage gap "
            "and its effect on confidence. Do not invent headlines, events, dates, "
            "or sources, and do not repeat this failed request during this analysis."
        )


def with_news_data_warning(report: str, unavailable_tools) -> str:
    """Persist a coverage notice even if the model omits it from its narrative."""
    tools = sorted(set(unavailable_tools) & {"get_news", "get_global_news"})
    if not tools:
        return report
    chinese = str(get_config().get("output_language", "English")).strip().lower() in {
        "chinese", "中文", "简体中文",
    }
    labels = {"get_news": "个股/标的新闻" if chinese else "asset-specific news",
              "get_global_news": "全球新闻" if chinese else "global news"}
    sources = ("、" if chinese else ", ").join(labels[tool] for tool in tools)
    notice = (
        f"> **数据缺失提示：** 本次未能获取部分新闻数据（{sources}）。"
        "分析仅基于其他可用资料，新闻相关判断的依据不足；数据缺失不代表没有新闻事件或情绪中性。"
        if chinese else
        f"> **Data availability:** Some news data could not be retrieved ({sources}). "
        "Analysis uses the remaining evidence, with limited support for news-related "
        "conclusions. Missing data does not imply no news events or neutral sentiment."
    )
    return f"{notice}\n\n{report}"


@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve news data for a given ticker symbol.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    return _get_optional_news("get_news", ticker, start_date, end_date)

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int | None, "Days to look back; omit to use the configured default"] = None,
    limit: Annotated[int | None, "Max articles to return; omit to use the configured default"] = None,
) -> str:
    """
    Retrieve global news data.
    Uses the configured news_data vendor. Defaults for look_back_days and
    limit come from DEFAULT_CONFIG (global_news_lookback_days,
    global_news_article_limit); pass explicit values to override.

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back; omit to inherit config
        limit (int): Maximum number of articles to return; omit to inherit config

    Returns:
        str: A formatted string containing global news data
    """
    return _get_optional_news("get_global_news", curr_date, look_back_days, limit)

@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A report of insider transaction data
    """
    return route_to_vendor("get_insider_transactions", ticker)
