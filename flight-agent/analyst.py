# analyst.py
import json
from typing import TypedDict, Optional, List


# ── 1. EXTENDED STATE ─────────────────────────────────────────
# We add the Analyst's outputs to the growing state.

class FlightSearchState(TypedDict):
    # From Planner
    user_input: str
    origin: Optional[str]
    destination: Optional[str]
    date: Optional[str]
    preference: Optional[str]
    max_stops: Optional[int]
    passengers: Optional[int]
    cabin_class: Optional[str]
    parsing_confidence: Optional[str]

    # From Researcher
    raw_flights: Optional[List[dict]]
    flight_count: Optional[int]
    search_attempts: Optional[int]
    data_source: Optional[str]

    # Analyst fills these ↓
    ranked_flights: Optional[List[dict]]   # Scored + sorted flights
    best_flight: Optional[dict]            # Single winner
    analysis_summary: Optional[str]        # Human-readable insight
    price_range: Optional[dict]            # min/max/avg for context

    error: Optional[str]


# ── 2. THE SCORING ENGINE ─────────────────────────────────────
# Pure Python. No LLM. Deliberate choice — explained above.

def normalize(values: List[float]) -> List[float]:
    """
    Scales a list of values to 0-1 range.
    
    Why min-max normalization specifically?
    Because we want relative comparison within this result set.
    The cheapest flight gets score 1.0, most expensive gets 0.0.
    Everything else proportionally in between.
    
    This means scores change based on the result set —
    which is correct. "Cheap" is relative to what's available.
    """
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        # All values identical — everyone gets 0.5
        # Why not 1.0? Because 0.5 signals "no differentiation"
        # rather than "everyone is best"
        return [0.5] * len(values)
    
    return [(v - min_val) / (max_val - min_val) for v in values]


def score_flights(flights: List[dict], preference: str) -> List[dict]:
    """
    Scores and ranks flights based on user preference.
    Returns flights with score added, sorted best first.
    """
    
    if not flights:
        return []
    
    # ── FILTER PASS ───────────────────────────────────────────
    # Remove flights with missing critical data
    # Why filter before scoring? Incomplete data corrupts
    # normalization — a ₹0 price would make everything else
    # look expensive. Garbage in, garbage out.
    
    valid_flights = [
        f for f in flights
        if f.get("price_inr", 0) > 0
        and f.get("duration_minutes", 0) > 0
    ]
    
    if not valid_flights:
        return []
    
    # ── PREFERENCE: CHEAPEST ──────────────────────────────────
    if preference == "cheapest":
        # Simple sort — no complex scoring needed
        # Why not score here? When the goal is purely cheapest,
        # scoring adds complexity without adding information.
        # Simpler is better when simpler is correct.
        sorted_flights = sorted(valid_flights, key=lambda f: f["price_inr"])
        for i, f in enumerate(sorted_flights):
            f["score"] = round(1.0 - (i / len(sorted_flights)), 2)
            f["score_reason"] = f"Rank #{i+1} by price"
        return sorted_flights

    # ── PREFERENCE: FASTEST ───────────────────────────────────
    elif preference == "fastest":
        sorted_flights = sorted(valid_flights, key=lambda f: f["duration_minutes"])
        for i, f in enumerate(sorted_flights):
            f["score"] = round(1.0 - (i / len(sorted_flights)), 2)
            f["score_reason"] = f"Rank #{i+1} by duration"
        return sorted_flights

    # ── PREFERENCE: FEWEST_STOPS ──────────────────────────────
    elif preference == "fewest_stops":
        sorted_flights = sorted(
            valid_flights,
            key=lambda f: (f["stops"], f["price_inr"])  # stops first, then price as tiebreaker
        )
        for i, f in enumerate(sorted_flights):
            f["score"] = round(1.0 - (i / len(sorted_flights)), 2)
            f["score_reason"] = f"Rank #{i+1} by stops"
        return sorted_flights

    # ── PREFERENCE: BEST_VALUE (Multi-criteria) ───────────────
    else:
        # This is the interesting case. Multi-criteria scoring.
        
        prices = [f["price_inr"] for f in valid_flights]
        durations = [f["duration_minutes"] for f in valid_flights]
        stops_list = [f["stops"] for f in valid_flights]
        
        # Normalize all three dimensions
        # Why invert price and duration scores?
        # Lower price = better, so cheapest gets 1.0 (inverted)
        # Lower duration = better, so fastest gets 1.0 (inverted)
        # Lower stops = better, so non-stop gets 1.0 (inverted)
        
        price_scores = [1 - s for s in normalize(prices)]
        duration_scores = [1 - s for s in normalize(durations)]
        stops_scores = [1 - s for s in normalize(stops_list)]
        
        # Weighted combination
        weights = {"price": 0.5, "duration": 0.3, "stops": 0.2}
        
        for i, flight in enumerate(valid_flights):
            composite_score = (
                price_scores[i] * weights["price"] +
                duration_scores[i] * weights["duration"] +
                stops_scores[i] * weights["stops"]
            )
            flight["score"] = round(composite_score, 3)
            flight["score_reason"] = (
                f"Price({price_scores[i]:.2f}) × 0.5 + "
                f"Speed({duration_scores[i]:.2f}) × 0.3 + "
                f"Stops({stops_scores[i]:.2f}) × 0.2"
            )
        
        # Sort by composite score descending
        return sorted(valid_flights, key=lambda f: f["score"], reverse=True)


def generate_analysis_summary(
    ranked_flights: List[dict],
    preference: str,
    origin: str,
    destination: str
) -> str:
    """
    Generates a human-readable insight about the results.
    
    Why generate this here and not with an LLM?
    This is template-based text, not reasoning.
    Template text is faster, cheaper, and more predictable.
    Save LLMs for tasks that actually need language understanding.
    """
    
    if not ranked_flights:
        return "No valid flights found for analysis."
    
    best = ranked_flights[0]
    cheapest = min(ranked_flights, key=lambda f: f["price_inr"])
    fastest = min(ranked_flights, key=lambda f: f["duration_minutes"])
    nonstops = [f for f in ranked_flights if f["stops"] == 0]
    
    hours = best["duration_minutes"] // 60
    mins = best["duration_minutes"] % 60
    
    summary_parts = [
        f"Found {len(ranked_flights)} flights from {origin} to {destination}.",
        f"Top pick ({preference}): {best['airline']} at ₹{best['price_inr']:,} — {hours}h {mins}m, {best['stops']} stop(s)."
    ]
    
    # Add contextual insights
    if best != cheapest:
        saving = best["price_inr"] - cheapest["price_inr"]
        summary_parts.append(
            f"Cheapest available: ₹{cheapest['price_inr']:,} ({cheapest['airline']}) "
            f"— ₹{saving:,} less but may take longer."
        )
    
    if best != fastest:
        time_diff = best["duration_minutes"] - fastest["duration_minutes"]
        summary_parts.append(
            f"Fastest option: {fastest['airline']} at {fastest['duration_minutes']//60}h "
            f"{fastest['duration_minutes']%60}m — {time_diff} mins quicker than top pick."
        )
    
    if nonstops:
        nonstop_prices = [f["price_inr"] for f in nonstops]
        summary_parts.append(
            f"Non-stop options available: {len(nonstops)} flights, "
            f"₹{min(nonstop_prices):,}–₹{max(nonstop_prices):,}."
        )
    else:
        summary_parts.append("No non-stop flights on this route today.")
    
    return " ".join(summary_parts)


# ── 3. THE ANALYST NODE ───────────────────────────────────────

def analyst_node(state: FlightSearchState) -> dict:
    """
    Scores, ranks, and derives insights from raw flight data.
    Pure Python — no LLM calls. Deterministic and fast.
    """
    
    raw_flights = state.get("raw_flights") or []
    preference = state.get("preference") or "cheapest"
    
    print(f"\n📊 Analyst: Processing {len(raw_flights)} flights")
    print(f"   Preference: {preference}")
    
    if not raw_flights:
        return {
            "ranked_flights": [],
            "best_flight": None,
            "analysis_summary": "No flights available to analyze.",
            "price_range": None,
            "error": "No raw flight data to analyze"
        }
    
    # Apply max_stops filter if user specified one
    # Why filter BEFORE scoring? Same reason as before —
    # excluded flights shouldn't influence normalization.
    
    max_stops = state.get("max_stops")
    if max_stops is not None:
        filtered = [f for f in raw_flights if f.get("stops", 99) <= max_stops]
        print(f"   Filtered to {len(filtered)} flights (max {max_stops} stops)")
    else:
        filtered = raw_flights
    
    # Score and rank
    ranked = score_flights(filtered, preference)
    
    if not ranked:
        return {
            "ranked_flights": [],
            "best_flight": None,
            "analysis_summary": "No flights matched your filters.",
            "price_range": None,
            "error": "All flights filtered out"
        }
    
    # Price statistics for context
    # Why compute this? The Critic uses price range to detect
    # anomalies — a ₹500 DEL-BOM ticket is suspicious.
    prices = [f["price_inr"] for f in ranked]
    price_range = {
        "min": min(prices),
        "max": max(prices),
        "avg": round(sum(prices) / len(prices)),
        "spread": max(prices) - min(prices)
    }
    
    print(f"   Price range: ₹{price_range['min']:,} – ₹{price_range['max']:,}")
    print(f"   Best pick: {ranked[0]['airline']} at ₹{ranked[0]['price_inr']:,}")
    
    summary = generate_analysis_summary(
        ranked,
        preference,
        state.get("origin", ""),
        state.get("destination", "")
    )
    
    return {
        "ranked_flights": ranked,
        "best_flight": ranked[0],
        "analysis_summary": summary,
        "price_range": price_range,
        "error": None
    }


# ── 4. ASSEMBLE THE 3-NODE GRAPH ──────────────────────────────

from Planner import planner_node
from researcher import researcher_node

from langgraph.graph import StateGraph, END

def build_graph():
    graph = StateGraph(FlightSearchState)
    
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", END)
    
    return graph.compile()


# ── 5. RUN ────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_graph()
    
    # Test 
    queries = [
        "cheapest flight from delhi to mumbai tomorrow",
        "fastest flight from bangalore to hyderabad this friday",
        "best value flight from delhi to goa next monday",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        
        result = app.invoke({
            "user_input": query,
            "origin": None, "destination": None,
            "date": None, "preference": None,
            "max_stops": None, "passengers": None,
            "cabin_class": None, "parsing_confidence": None,
            "raw_flights": None, "flight_count": None,
            "search_attempts": None, "data_source": None,
            "ranked_flights": None, "best_flight": None,
            "analysis_summary": None, "price_range": None,
            "error": None
        })
        
        print(f"\n📋 ANALYSIS SUMMARY:")
        print(result["analysis_summary"])
        
        print(f"\n🏆 TOP 3 FLIGHTS:")
        for flight in (result.get("ranked_flights") or [])[:3]:
            hours = flight["duration_minutes"] // 60
            mins = flight["duration_minutes"] % 60
            print(f"\n  {flight['airline']} {flight['flight_number']}")
            print(f"  ₹{flight['price_inr']:,} | {hours}h {mins}m | {flight['stops']} stop(s)")
            print(f"  Score: {flight['score']} — {flight['score_reason']}")