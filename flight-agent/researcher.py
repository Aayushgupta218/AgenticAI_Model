# researcher.py
import os
import json
from pathlib import Path
from typing import TypedDict, Optional, List
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from serpapi import GoogleSearch
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# load_dotenv()
load_dotenv(Path(__file__).with_name(".env"))

# ── 1. EXTENDED STATE ─────────────────────────────────────────
# We ADD to the state from Phase 1, never replace it.
# This is the conveyor belt growing as more agents add to it.

class FlightSearchState(TypedDict):
    # From Planner (Phase 1)
    user_input: str
    origin: Optional[str]
    destination: Optional[str]
    date: Optional[str]
    preference: Optional[str]
    max_stops: Optional[int]
    passengers: Optional[int]
    cabin_class: Optional[str]
    parsing_confidence: Optional[str]

    # Researcher fills these ↓
    raw_flights: Optional[List[dict]]     # Everything the API returned
    flight_count: Optional[int]           # How many results we got
    search_attempts: Optional[int]        # How many times we retried
    data_source: Optional[str]            # Which API gave us the data

    error: Optional[str]


# ── 2. THE TOOLS ──────────────────────────────────────────────
# @tool decorator is how LangGraph/LangChain knows this function
# is available for an LLM to call. The docstring is CRITICAL —
# the LLM reads it to understand when and how to use the tool.
# Bad docstring = LLM misuses or ignores the tool.

@tool
def search_google_flights(
    origin: str,
    destination: str,
    date: str,
    passengers: int = 1,
    cabin_class: str = "economy"
) -> str:
    """
    Search for flights using Google Flights via SerpAPI.
    Use this when you have valid IATA codes and a date.
    
    Args:
        origin: IATA code of departure airport (e.g. 'DEL')
        destination: IATA code of arrival airport (e.g. 'BOM')
        date: Date in YYYY-MM-DD format
        passengers: Number of adult passengers (default 1)
        cabin_class: 'economy', 'business', or 'first'
    
    Returns:
        JSON string with flight results or error message
    """
    
    # SerpAPI cabin class mapping
    # Why do this mapping? API expects specific integer codes,
    # not strings. This translation layer is YOUR job as engineer.
    cabin_map = {
        "economy": 1,
        "business": 2, 
        "first": 3
    }
    
    params = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": date,
        "adults": passengers,
        "travel_class": cabin_map.get(cabin_class, 1),
        "currency": "INR",
        "hl": "en",
        "api_key": os.getenv("SERPAPI_KEY")
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Why check this key specifically?
        # SerpAPI returns error messages inside the JSON, not as
        # HTTP errors. A 200 response can still be a failed search.
        if "error" in results:
            return json.dumps({
                "success": False,
                "error": results["error"],
                "flights": []
            })
        
        # Extract best_flights (Google's top picks) and
        # other_flights (remaining options)
        # Why both? Best flights alone is only 3-5 results.
        # We want the full picture for the Analyst.
        
        best = results.get("best_flights", [])
        other = results.get("other_flights", [])
        all_flights = best + other
        
        if not all_flights:
            return json.dumps({
                "success": False,
                "error": "No flights found for this route and date",
                "flights": []
            })
        
        # Why clean/normalize here and not in Analyst?
        # The Analyst should receive consistent data regardless
        # of which API provided it. Normalization at the source
        # is the "adapter pattern" — a real software design principle.
        
        cleaned_flights = []
        for flight in all_flights:
            # Each flight has "flights" array (legs) inside it
            legs = flight.get("flights", [])
            if not legs:
                continue
                
            first_leg = legs[0]
            last_leg = legs[-1]
            
            cleaned_flights.append({
                "airline": first_leg.get("airline", "Unknown"),
                "flight_number": first_leg.get("flight_number", ""),
                "departure_time": first_leg.get("departure_airport", {}).get("time", ""),
                "arrival_time": last_leg.get("arrival_airport", {}).get("time", ""),
                "duration_minutes": flight.get("total_duration", 0),
                "stops": len(legs) - 1,
                "price_inr": flight.get("price", 0),
                "is_best_flight": flight in best,
                "layovers": [
                    leg.get("arrival_airport", {}).get("name", "")
                    for leg in legs[:-1]  # All legs except last = layover points
                ]
            })
        
        print(f"\n✅ Researcher: Found {len(cleaned_flights)} flights")
        
        return json.dumps({
            "success": True,
            "flights": cleaned_flights,
            "count": len(cleaned_flights)
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "flights": []
        })


@tool  
def validate_iata_code(city_name: str) -> str:
    """
    Converts a city name to its IATA airport code.
    Use this ONLY if origin or destination is a city name, not already an IATA code.
    IATA codes are exactly 3 uppercase letters (e.g. DEL, BOM, BLR).
    
    Args:
        city_name: Name of the city (e.g. 'Delhi', 'Mumbai')
    
    Returns:
        JSON with iata_code or error
    """
    
    # Why a hardcoded map and not another API call?
    # For a portfolio project, this is pragmatic.
    # In production you'd use an airport database.
    # The principle: don't add dependencies you don't need.
    
    iata_map = {
        "delhi": "DEL", "new delhi": "DEL",
        "mumbai": "BOM", "bombay": "BOM",
        "bangalore": "BLR", "bengaluru": "BLR",
        "hyderabad": "HYD", "chennai": "MAA", "madras": "MAA",
        "kolkata": "CCU", "calcutta": "CCU",
        "pune": "PNQ", "ahmedabad": "AMD",
        "goa": "GOI", "jaipur": "JAI",
        "lucknow": "LKO", "kochi": "COK", "cochin": "COK",
        "chandigarh": "IXC", "amritsar": "ATQ",
        "varanasi": "VNS", "bhopal": "BHO",
        "nagpur": "NAG", "indore": "IDR",
        "srinagar": "SXR", "leh": "IXL",
        "port blair": "IXZ", "agartala": "IXA"
    }
    
    code = iata_map.get(city_name.lower().strip())
    
    if code:
        return json.dumps({"success": True, "iata_code": code})
    else:
        return json.dumps({
            "success": False,
            "error": f"Could not find IATA code for '{city_name}'"
        })


# ── 3. THE RESEARCHER NODE ────────────────────────────────────
# This is where the ReAct loop lives. We give the LLM tools
# and let it decide what to call and when.

tools = [search_google_flights, validate_iata_code]

def researcher_node(state: FlightSearchState) -> dict:
    """
    Researches actual flight options using available tools.
    Runs a ReAct loop internally until it has sufficient results.
    """
    
    # Why bind tools to the LLM specifically?
    # This tells the LLM "these are the functions you can call"
    # It adds tool schemas to the LLM's context automatically.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",        
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    ).bind_tools(tools)
    
    system_prompt = """You are a flight data researcher. Your job is to find real flight options.

You have access to:
1. search_google_flights — searches real flight data
2. validate_iata_code — converts city names to airport codes if needed

Steps to follow:
1. Check if origin and destination are valid 3-letter IATA codes
   - If not, use validate_iata_code first
2. Call search_google_flights with the parameters
3. If results are empty or less than 3 flights, try once more
4. Return what you found

Be systematic. Always verify you have IATA codes before searching."""

    # Build the user message from state
    # Why construct this message dynamically?
    # The LLM needs context from the Planner's output.
    # We're passing the structured state as natural language
    # so the LLM understands what it's working with.
    
    user_message = f"""Find flights with these parameters:
- From: {state.get('origin')} 
- To: {state.get('destination')}
- Date: {state.get('date')}
- Passengers: {state.get('passengers', 1)}
- Cabin: {state.get('cabin_class', 'economy')}
- Preference: {state.get('preference', 'cheapest')}

Search for available flights and return the results."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # ── THE ReAct LOOP ────────────────────────────────────────
    # Why a loop with max_iterations?
    # Without a limit, a confused agent can loop forever.
    # max_iterations is your circuit breaker.
    # 5 is enough: validate + search + retry + 2 buffer.
    
    max_iterations = 5
    iterations = 0
    search_attempts = 0
    all_flights = []

    print("SERPAPI key present:", bool(os.getenv("SERPAPI_KEY")))
    
    while iterations < max_iterations:
        iterations += 1

        response = llm.invoke(messages)

        tool_calls = getattr(response, "tool_calls", None) or []

        messages.append(AIMessage(content=response.content, tool_calls=tool_calls))

        if not tool_calls:
            print(f"\n✅ Researcher finished after {iterations} iterations")
            break

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"\n🔧 Calling tool: {tool_name}")
            print(f"   Args: {tool_args}")

            tool_func = next((t for t in tools if t.name == tool_name), None)
            if not tool_func:
                tool_result = json.dumps({"error": f"Tool {tool_name} not found"})
            else:
                tool_result = tool_func.invoke(tool_args)

                if tool_name == "search_google_flights":
                    search_attempts += 1
                    result_data = json.loads(tool_result)
                    if result_data.get("success") and result_data.get("flights"):
                        all_flights = result_data["flights"]

            messages.append(ToolMessage(
                content=tool_result,
                tool_call_id=tool_call.get("id", "tool"),
                name=tool_name
            ))
    
    # Write results to state
    if all_flights:
        return {
            "raw_flights": all_flights,
            "flight_count": len(all_flights),
            "search_attempts": search_attempts,
            "data_source": "google_flights_serpapi",
            "error": None
        }
    else:
        return {
            "raw_flights": [],
            "flight_count": 0,
            "search_attempts": search_attempts,
            "data_source": "google_flights_serpapi",
            "error": "No flights found after exhausting search attempts"
        }


# ── 4. IMPORT PLANNER FROM PHASE 1 ───────────────────────────

from Planner import planner_node   # Reuse exactly what you built


# ── 5. BUILD THE 2-NODE GRAPH ─────────────────────────────────

def build_graph():
    graph = StateGraph(FlightSearchState)
    
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    
    # Why this edge order?
    # Planner must finish before Researcher has data to work with.
    # Sequential dependency = sequential edges.
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", END)
    
    return graph.compile()


# ── 6. RUN IT ─────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_graph()
    
    query = "cheapest flights from delhi to mumbai tomorrow"
    
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
        "error": None
    })
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {result['flight_count']} flights found")
    print(f"Source: {result['data_source']}")
    print(f"Search attempts: {result['search_attempts']}")
    print(f"\nSample flights:")
    
    for flight in (result.get("raw_flights") or [])[:3]:
        print(f"\n  ✈ {flight['airline']} {flight['flight_number']}")
        print(f"    {flight['departure_time']} → {flight['arrival_time']}")
        print(f"    Stops: {flight['stops']} | Price: ₹{flight['price_inr']:,}")