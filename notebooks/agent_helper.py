# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Financial research agent used by 09_single_agent_single_turn_evaluation_demo.ipynb.

A Strands agent with three market-data tools backed by fixed mock data, so the
ground truth in data/single_turn_financial.csv is deterministic. The agent is
invoked in-process (no HTTP server); each call returns the native Strands
telemetry that StrandsAdapter.transform_to_canonical() expects.

The model defaults to UAEF_LLM_MODEL_ID (or Claude Sonnet 4.6 on Bedrock) and
uses the standard AWS credential chain and AWS_REGION.
"""

import os
from typing import Any, Dict, Optional

from strands import Agent, tool
from strands.models import BedrockModel

_PRICES = {
    "AMZN": {"ticker": "AMZN", "price": 185.20, "currency": "USD", "change_pct": 1.4},
    "AAPL": {"ticker": "AAPL", "price": 227.50, "currency": "USD", "change_pct": -0.6},
    "MSFT": {"ticker": "MSFT", "price": 431.10, "currency": "USD", "change_pct": 0.3},
    "NVDA": {"ticker": "NVDA", "price": 118.90, "currency": "USD", "change_pct": 2.1},
}

_NEWS = {
    "AMZN": ["Amazon expands same-day delivery to 20 new cities"],
    "AAPL": ["Apple announces new on-device AI features"],
    "MSFT": ["Microsoft reports 16% growth in cloud revenue"],
    "NVDA": ["NVIDIA unveils next-generation data center GPU"],
}

_RATIOS = {
    "AMZN": {"ticker": "AMZN", "pe_ratio": 42.3, "debt_to_equity": 0.6},
    "AAPL": {"ticker": "AAPL", "pe_ratio": 34.8, "debt_to_equity": 1.9},
    "MSFT": {"ticker": "MSFT", "pe_ratio": 36.1, "debt_to_equity": 0.3},
    "NVDA": {"ticker": "NVDA", "pe_ratio": 55.7, "debt_to_equity": 0.2},
}

SYSTEM_PROMPT = (
    "You are a financial research assistant. Always use the provided tools to look up "
    "stock prices, company news, and financial ratios; never guess figures. Call only the "
    "tools needed to answer the question. If a tool reports that no data is available, say "
    "so plainly. Answer concisely."
)


def _unknown(ticker: str) -> Dict[str, str]:
    return {"error": f"No data available for {ticker.upper()}"}


@tool
def get_stock_price(ticker: str) -> dict:
    """Get the latest stock price for a ticker symbol (e.g. "AMZN")."""
    return _PRICES.get(ticker.upper(), _unknown(ticker))


@tool
def get_company_news(ticker: str) -> dict:
    """Get the latest news headlines for a company by ticker symbol."""
    headlines = _NEWS.get(ticker.upper())
    return {"ticker": ticker.upper(), "headlines": headlines} if headlines else _unknown(ticker)


@tool
def get_financial_ratios(ticker: str) -> dict:
    """Get key financial ratios (P/E, debt-to-equity) for a ticker symbol."""
    return _RATIOS.get(ticker.upper(), _unknown(ticker))


TOOLS = [get_stock_price, get_company_news, get_financial_ratios]

_model: Optional[BedrockModel] = None


def start_agent() -> None:
    """Configure the Bedrock model used by every invocation."""
    global _model
    model_id = os.getenv("UAEF_LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    _model = BedrockModel(model_id=model_id, temperature=0.0)
    print(f"✓ Financial research agent ready (model: {model_id})")


def stop_agent() -> None:
    """Release the model; safe to call more than once."""
    global _model
    _model = None


def invoke_agent(query: str, session_id: str) -> Dict[str, Any]:
    """Run one query on a fresh agent and return Strands telemetry for StrandsAdapter."""
    if _model is None:
        start_agent()
    # A fresh agent per query keeps single-turn test cases independent
    agent = Agent(model=_model, tools=TOOLS, system_prompt=SYSTEM_PROMPT, callback_handler=None)
    result = agent(query)
    return {
        "messages": agent.messages,
        "metrics_summary": agent.event_loop_metrics.get_summary(),
        "stop_reason": result.stop_reason,
        "session_id": session_id,
    }
