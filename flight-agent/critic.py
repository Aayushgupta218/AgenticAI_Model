import json
from typing import TypedDict, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()


# ── 1. FINAL STATE ────────────────────────────────────────────
# Two new fields: critic_verdict and retry_count
# retry_count is the circuit breaker — prevents infinite loops

class FlightSearchState(TypedDict):
    # Planner
    user_input: str
    origin: Optional[str]
    destination: Optional[str]
    date: Optional[str]
    preference: Optional[str]
    max_stops: Optional[int]
    passengers: Optional[int]
    cabin_class: Optional[str]
    parsing_confidence: Optional[str]

    # Researcher
    raw_flights: Optional[List[dict]]
    flight_count: Optional[int]
    search_attempts: Optional[int]
    data_source: Optional[str]

    # Analyst
    ranked_flights: Optional[List[dict]]
    best_flight: Optional[dict]
    analysis_summary: Optional[str]
    price_range: Optional[dict]

    # Critic fills these ↓
    critic_verdict: Optional[str]      # "approved" | "retry" | "needs_clarity"
    critic_reasoning: Optional[str]    # Why it made this decision
    retry_count: Optional[int]         # Circuit breaker
    final_output: Optional[dict]       # Clean output for the user

    error: Optional[str]


# ── 2. SANITY BOUNDS ──────────────────────────────────────────
# Hardcoded price bounds for Indian domestic routes.

ROUTE_PRICE_BOUNDS = {
    "domestic_india": {"min": 1500, "max": 50000},
    "short_haul": {"min": 1500, "max": 20000},   # < 2 hrs
    "medium_haul": {"min": 2000, "max": 35000},  # 2-4 hrs
    "long_haul": {"min": 3000, "max": 50000},    # > 4 hrs
}

def get_route_bounds(duration_minutes: int) -> dict:
    """Returns price bounds based on flight duration."""
    if duration_minutes < 120:
        return ROUTE_PRICE_BOUNDS["short_haul"]
    elif duration_minutes < 240:
        return ROUTE_PRICE_BOUNDS["medium_haul"]
    else:
        return ROUTE_PRICE_BOUNDS["long_haul"]


# ── 3. THE CRITIC'S CHECKS ────────────────────────────────────

def check_price_sanity(ranked_flights: List[dict]) -> tuple[bool, str]:
    """
    Returns (is_sane, reason).
    Checks if prices fall within realistic bounds.
    """
    if not ranked_flights:
        return False, "No flights to validate"
    
    for flight in ranked_flights[:3]:  # Check top 3 only
        price = flight.get("price_inr", 0)
        duration = flight.get("duration_minutes", 120)
        bounds = get_route_bounds(duration)
        
        if price < bounds["min"]:
            return False, (
                f"Suspiciously low price: ₹{price:,} for "
                f"{duration}min flight. Minimum expected: ₹{bounds['min']:,}"
            )
        if price > bounds["max"]:
            return False, (
                f"Suspiciously high price: ₹{price:,}. "
                f"Maximum expected: ₹{bounds['max']:,}"
            )
    
    return True, "Prices within expected range"


def check_result_sufficiency(flight_count: int) -> tuple[bool, str]:
    """
    Returns (is_sufficient, reason).
    We need at least 3 flights to make a meaningful recommendation.
    """
    if flight_count >= 5:
        return True, f"Good result set: {flight_count} flights"
    elif flight_count >= 3:
        return True, f"Acceptable result set: {flight_count} flights"
    elif flight_count > 0:
        return False, f"Thin result set: only {flight_count} flight(s) found"
    else:
        return False, "No flights found"


def check_parsing_confidence(
    parsing_confidence: str,
    origin: str,
    destination: str,
    date: str
) -> tuple[bool, str]:
    """
    Returns (is_confident, reason).
    Low confidence + missing fields = ask user to clarify.
    """
    if parsing_confidence == "low":
        missing = []
        if not origin:
            missing.append("departure city")
        if not destination:
            missing.append("destination city")
        if not date:
            missing.append("travel date")
        
        if missing:
            return False, f"Unclear query — missing: {', '.join(missing)}"
        else:
            return False, "Query was ambiguous — results may not match intent"
    
    return True, f"Parsing confidence: {parsing_confidence}"


def check_data_consistency(ranked_flights: List[dict]) -> tuple[bool, str]:
    """
    Returns (is_consistent, reason).
    Catches obvious data corruption — duplicate flights, 
    zero durations, impossible times.
    """
    if not ranked_flights:
        return False, "Empty flight list"
    
    # Check for duplicate flight numbers
    flight_numbers = [
        f.get("flight_number", "") 
        for f in ranked_flights 
        if f.get("flight_number")
    ]
    if len(flight_numbers) != len(set(flight_numbers)):
        return False, "Duplicate flight numbers detected — possible data corruption"
    
    # Check for zero/negative durations
    for f in ranked_flights:
        if f.get("duration_minutes", 1) <= 0:
            return False, f"Invalid duration for {f.get('flight_number', 'unknown flight')}"
    
    return True, "Data consistency checks passed"


# ── 4. THE CRITIC NODE ────────────────────────────────────────
# Uses LLM for ONE specific thing: synthesizing multiple check
# results into a final verdict with clear reasoning.

def critic_node(state: FlightSearchState) -> dict:
    """
    Validates the pipeline's output. Returns verdict + reasoning.
    Verdict determines graph routing: approved/retry/needs_clarity.
    """
    
    ranked_flights = state.get("ranked_flights") or []
    flight_count = state.get("flight_count") or 0
    retry_count = state.get("retry_count") or 0
    
    print(f"\n🔍 Critic: Evaluating results (attempt #{retry_count + 1})")
    
    # ── CIRCUIT BREAKER ───────────────────────────────────────
    # Why check this first, before anything else?
    # If we've already retried twice and still have bad results,
    # the problem isn't fixable by retrying — it's the route,
    # the date, or the API. Stop and tell the user honestly.
    
    if retry_count >= 2:
        print("   ⚠️  Max retries reached — approving best available")
        
        if ranked_flights:
            verdict = "approved"
            reasoning = (
                f"Approved after {retry_count} retries. "
                f"Results may not be ideal but are the best available. "
                f"Found {flight_count} flights."
            )
        else:
            verdict = "needs_clarity"
            reasoning = (
                "After multiple attempts, no valid flights found. "
                "Please verify the route and date."
            )
        
        return {
            "critic_verdict": verdict,
            "critic_reasoning": reasoning,
            "retry_count": retry_count,
            "final_output": build_final_output(state) if ranked_flights else None
        }
    
    # ── RUN ALL CHECKS ────────────────────────────────────────
    checks = {}
    
    checks["price_sanity"] = check_price_sanity(ranked_flights)
    checks["sufficiency"] = check_result_sufficiency(flight_count)
    checks["parsing"] = check_parsing_confidence(
        state.get("parsing_confidence", "medium"),
        state.get("origin", ""),
        state.get("destination", ""),
        state.get("date", "")
    )
    checks["consistency"] = check_data_consistency(ranked_flights)
    
    # Log check results
    for check_name, (passed, reason) in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}: {reason}")
    
    # ── LLM VERDICT ───────────────────────────────────────────
    # Now use LLM to synthesize check results into a verdict.
    # Give it the check results + context, ask for judgment.
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    
    check_summary = "\n".join([
        f"- {name}: {'PASS' if passed else 'FAIL'} — {reason}"
        for name, (passed, reason) in checks.items()
    ])
    
    system_prompt = """You are a quality control agent for a flight search system.
    
You receive results from validation checks and must decide on ONE verdict:

APPROVED    — results are good enough to show the user
RETRY       — results have fixable issues (too few results, API hiccup) — search again  
NEEDS_CLARITY — query was too ambiguous or route is invalid — must ask user

Rules:
- If parsing FAILED with missing fields → NEEDS_CLARITY (can't fix by retrying)
- If sufficiency FAILED → RETRY (might get more results with another attempt)  
- If price_sanity FAILED → RETRY (might be a data glitch)
- If consistency FAILED → RETRY (data corruption, try fresh)
- If ALL checks pass → APPROVED
- If multiple failures → use judgment, lean toward NEEDS_CLARITY if user intent unclear

Respond with ONLY a JSON object:
{
    "verdict": "approved" | "retry" | "needs_clarity",
    "reasoning": "one sentence explanation",
    "primary_issue": "the main problem if not approved, else null"
}"""

    user_message = f"""Check results:
{check_summary}

Context:
- Flights found: {flight_count}
- Retry count so far: {retry_count}
- Route: {state.get('origin')} → {state.get('destination')}
- Date: {state.get('date')}

What is your verdict?"""

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ])
    
    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        
        verdict_data = json.loads(raw.strip())
        verdict = verdict_data.get("verdict", "approved")
        reasoning = verdict_data.get("reasoning", "")
        
    except json.JSONDecodeError:
        # If LLM fails to return valid JSON, default to approved
        # Why approved and not retry? Failing safe — better to show
        # imperfect results than loop forever on a parse error.
        verdict = "approved"
        reasoning = "Critic parse error — defaulting to approved"
    
    print(f"\n   🏛️  Verdict: {verdict.upper()}")
    print(f"   Reasoning: {reasoning}")
    
    return {
        "critic_verdict": verdict,
        "critic_reasoning": reasoning,
        "retry_count": retry_count + 1,
        "final_output": build_final_output(state) if verdict == "approved" else None,
        "error": None
    }


# ── 5. FINAL OUTPUT BUILDER ───────────────────────────────────
# Why a separate function for this?
# Clean separation: Critic decides IF output is good,
# this function decides HOW output is formatted.
# Single responsibility principle.

def build_final_output(state: FlightSearchState) -> dict:
    """Builds the clean, user-facing output dict."""
    
    ranked = state.get("ranked_flights") or []
    best = state.get("best_flight") or {}
    
    return {
        "query": state.get("user_input"),
        "route": f"{state.get('origin')} → {state.get('destination')}",
        "date": state.get("date"),
        "summary": state.get("analysis_summary"),
        "best_flight": {
            "airline": best.get("airline"),
            "flight_number": best.get("flight_number"),
            "price": f"₹{best.get('price_inr', 0):,}",
            "duration": f"{best.get('duration_minutes', 0) // 60}h {best.get('duration_minutes', 0) % 60}m",
            "stops": best.get("stops"),
            "departure": best.get("departure_time"),
            "arrival": best.get("arrival_time"),
            "score": best.get("score")
        },
        "all_options": [
            {
                "rank": i + 1,
                "airline": f.get("airline"),
                "price": f"₹{f.get('price_inr', 0):,}",
                "duration": f"{f.get('duration_minutes', 0) // 60}h {f.get('duration_minutes', 0) % 60}m",
                "stops": f.get("stops"),
                "score": f.get("score")
            }
            for i, f in enumerate(ranked[:5])  # Top 5 only
        ],
        "price_range": state.get("price_range"),
        "critic_approved": True,
        "data_source": state.get("data_source")
    }


# ── 6. THE ROUTING FUNCTION ───────────────────────────────────
# This is what conditional edges call.
# It reads state and returns a string key.
# LangGraph maps that key to the next node.
# Pure routing logic — no side effects.

def route_after_critic(state: FlightSearchState) -> str:
    """
    Reads critic_verdict from state and returns routing key.
    This function is called by LangGraph after critic_node runs.
    """
    verdict = state.get("critic_verdict", "approved")
    print(f"\n🔀 Router: directing to '{verdict}'")
    return verdict


# ── 7. FULL 4-NODE GRAPH WITH CONDITIONAL EDGES ───────────────

from Planner import planner_node
from researcher import researcher_node
from analyst import analyst_node

def build_graph():
    graph = StateGraph(FlightSearchState)
    
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    
    graph.set_entry_point("planner")
    
    # Linear edges for first three nodes
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "critic")
    
    # THE CONDITIONAL EDGE — this is what makes it agentic
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "approved": END,
            "retry": "researcher",        # ← loop back
            "needs_clarity": END
        }
    )
    
    return graph.compile()


# ── 8. RUN IT ─────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_graph()
    
    query = "cheapest flight from delhi to Bangalore tomorrow"
    
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")
    
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
        "critic_verdict": None, "critic_reasoning": None,
        "retry_count": 0, "final_output": None,
        "error": None
    })
    
    print(f"\n{'='*60}")
    print(f"CRITIC VERDICT: {result.get('critic_verdict', '').upper()}")
    print(f"REASONING: {result.get('critic_reasoning')}")
    
    if result.get("final_output"):
        output = result["final_output"]
        print(f"\n🏆 BEST FLIGHT:")
        best = output["best_flight"]
        print(f"   {best['airline']} {best['flight_number']}")
        print(f"   {best['price']} | {best['duration']} | {best['stops']} stop(s)")
        print(f"   {best['departure']} → {best['arrival']}")
        print(f"\n📋 SUMMARY: {output['summary']}")
        print(f"\n📊 ALL OPTIONS:")
        for opt in output["all_options"]:
            print(f"   #{opt['rank']} {opt['airline']} — {opt['price']} | {opt['duration']} | score: {opt['score']}")
    else:
        print(f"\n⚠️  No output — verdict requires user action")
        print(f"   {result.get('critic_reasoning')}")