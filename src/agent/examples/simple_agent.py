"""Simple example agent — demonstrates the full prediction flow.

Gets market data and news, blends the signals, returns a probability.
No structured reasoning steps — just natural code. The validator
handles trace construction.
"""

from __future__ import annotations

from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult


class SimpleAgent(BaseAgent):
    agent_id = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
    agent_version = "0.1.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        market_slug = ctx.event_title.lower().replace(" ", "-")

        # Pull market data
        market = ctx.call_provider("polymarket", "get_market", {
            "market_slug": market_slug,
        })
        market_price = market.get("price", 0.5)
        volume = market.get("volume_24h", 0)

        # Pull recent news
        news = ctx.call_provider("web_search", "search", {
            "query": ctx.event_title,
            "num_results": 3,
        })
        snippets = [r.get("snippet", "") for r in news.get("results", [])]

        # Simple sentiment scoring from news snippets
        bullish_keywords = ["increased likelihood", "momentum", "adjusted positions"]
        bearish_keywords = ["uncertain", "barriers", "divided"]

        sentiment = 0.0
        for snippet in snippets:
            lower = snippet.lower()
            for kw in bullish_keywords:
                if kw in lower:
                    sentiment += 0.03
            for kw in bearish_keywords:
                if kw in lower:
                    sentiment -= 0.02

        # Blend: 80% market, 20% sentiment-adjusted prior
        blended = market_price * 0.8 + (0.5 + sentiment) * 0.2
        final_prob = max(0.01, min(0.99, blended))

        reasoning = (
            f"Polymarket price: {market_price:.3f} (24h vol: {volume:,.0f}). "
            f"Web search returned {len(snippets)} results with net sentiment "
            f"{sentiment:+.3f}. "
            f"Blended 80/20 market+sentiment → {final_prob:.3f}."
        )

        return AgentResult(
            prediction=round(final_prob, 4),
            reasoning=reasoning,
        )
