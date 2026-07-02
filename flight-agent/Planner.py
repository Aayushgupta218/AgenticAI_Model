from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import json
import os

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))


# ── 1. THE STATE ──────────────────────────────────────────────
# This is the shared conveyor belt. Every agent reads from and
# writes to this. Notice it holds RAW input AND structured output

class FlightSearchState(TypedDict):
    # Input
    user_input: str
    
    # Planner fills these
    origin: Optional[str]         # IATA code e.g. "DEL"
    destination: Optional[str]    # IATA code e.g. "BOM"
    date: Optional[str]           # ISO format "YYYY-MM-DD"
    preference: Optional[str]     # "cheapest", "fastest", "fewest_stops"
    max_stops: Optional[int]      # None means no preference
    passengers: Optional[int]
    cabin_class: Optional[str]
    
    # Planner's confidence in its own parsing
    # Why store this? Critic will use it later to decide
    # whether to trust the output or ask user to clarify
    parsing_confidence: Optional[str]  # "high", "medium", "low"
    
    # Error handling — why? Because agents fail.
    # We need a way to propagate errors through the graph
    # without crashing the whole system
    error: Optional[str]


# ── 2. THE PLANNER NODE ───────────────────────────────────────
# A "node" in LangGraph is just a function that takes State
# and returns a PARTIAL state update. You don't return the
# whole state — just the keys you're changing. LangGraph
# merges it automatically. This is important: nodes are
# surgeons, not replacements.

def planner_node(state: FlightSearchState) -> dict:
    """
    Parses natural language flight query into structured params.
    Returns only the keys it's responsible for filling.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        max_retries=2
    )

    today_day = datetime.now().strftime("%A, %B %d %Y")

    system_prompt = f"""You are a flight query parser. Today is {today_day}.

Your ONLY job is to extract structured flight search parameters from natural language.
Return ONLY a valid JSON object. No explanation. No markdown. No extra text.

JSON Schema you must follow exactly:
{{
    "origin": "IATA airport code (3 letters uppercase) or null if unclear",
    "destination": "IATA airport code (3 letters uppercase) or null if unclear",
    "date": "YYYY-MM-DD format, resolve relative dates like 'tomorrow', 'friday' etc or null",
    "preference": "one of: cheapest / fastest / fewest_stops / best_value — infer from context, default cheapest",
    "max_stops": "integer or null (null means no preference)",
    "passengers": "integer, default 1",
    "cabin_class": "one of: economy / business / first — default economy",
    "parsing_confidence": "high if all key fields clear, medium if some inferred, low if guessing"
}}

Common Indian city → IATA mappings you must know:
Delhi → DEL, Mumbai → BOM, Bangalore → BLR, Chennai → MAA,
Hyderabad → HYD, Kolkata → CCU, Pune → PNQ, Ahmedabad → AMD,
Goa → GOI, Jaipur → JAI, Lucknow → LKO, Kochi → COK
"""

    user_prompt = f"Parse this flight query: {state['user_input']}"

    try:
        response = llm.invoke([
            ("system", system_prompt),
            ("human", user_prompt)
        ])

        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw.strip())

        print("\n✅ Planner parsed successfully:")
        print(json.dumps(parsed, indent=2))

        return {
            "origin": parsed.get("origin"),
            "destination": parsed.get("destination"),
            "date": parsed.get("date"),
            "preference": parsed.get("preference", "cheapest"),
            "max_stops": parsed.get("max_stops"),
            "passengers": parsed.get("passengers", 1),
            "cabin_class": parsed.get("cabin_class", "economy"),
            "parsing_confidence": parsed.get("parsing_confidence", "medium"),
            "error": None
        }

    except json.JSONDecodeError as e:
        print(f"\n❌ Planner failed to parse JSON: {e}")
        return {"error": f"Planner JSON parse failed: {str(e)}"}
    except Exception as e:
        print(f"\n❌ Planner failed with unexpected error: {e}")
        return {"error": f"Planner failed: {str(e)}"}
    

# ── 3. THE GRAPH ──────────────────────────────────────────────
# This is where LangGraph comes in. We're defining the
# STRUCTURE of the pipeline — what runs, in what order,
# with what connections. 

def build_graph():
    graph = StateGraph(FlightSearchState)
    
    # Add our single node
    graph.add_node("planner", planner_node)
    
    # Set entry point — where does execution start?
    graph.set_entry_point("planner")
    
    # For now, planner → END. Next week: planner → researcher
    graph.add_edge("planner", END)
    
    # Compile = validate the graph structure + create runnable
    return graph.compile()


# ── 4. RUN IT ──────────────────────
if __name__ == "__main__":
    app = build_graph()
    
    # Test with messy, real-world style inputs
    test_queries = [
        "cheapest flight from delhi to mumbai tomorrow",
        "del to goa this friday, 2 people",
        "i need to fly bangalore to hyderabad next monday, business class",
        "flights to chennai from pune",   # No date — confidence should be low
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"INPUT: {query}")
        
        result = app.invoke({
            "user_input": query,
            "origin": None,
            "destination": None,
            "date": None,
            "preference": None,
            "max_stops": None,
            "passengers": None,
            "cabin_class": None,
            "parsing_confidence": None,
            "error": None
        })
        
        if result.get("error"):
            print(f"ERROR: {result['error']}")
        else:
            print(f"CONFIDENCE: {result['parsing_confidence']}")