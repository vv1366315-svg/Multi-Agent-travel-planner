import os
import certifi
import uuid
import operator

from typing import TypedDict, Annotated
from dotenv import load_dotenv

# ==========================================
# ENVIRONMENT CONFIGURATION
# ==========================================

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ==========================================
# IMPORTS
# ==========================================

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


# ==========================================
# API KEY CONFIGURATION
# ==========================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Please add it to your .env file."
    )


# ==========================================
# DATABASE CONFIGURATION
# ==========================================

def get_database_url():

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:

        separator = "&" if "?" in database_url else "?"

        database_url = (
            f"{database_url}{separator}sslmode=require"
        )

    return database_url


# ==========================================
# LLM CONFIGURATION
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)


# ==========================================
# TRAVEL STATE
# ==========================================

class TravelState(TypedDict):

    messages: Annotated[list[AnyMessage], operator.add]

    user_query: str

    flight_results: str

    hotel_results: str

    itinerary: str

    llm_calls: int


# ==========================================
# FLIGHT AGENT
# ==========================================

def flight_agent(state: TravelState):

    print("\n[Flight Agent] Searching for flights...")

    query = state["user_query"]

    flight_data = search_flights(query)

    return {

        "flight_results": flight_data,

        "messages": [
            AIMessage(
                content="Flight search completed."
            )
        ]

    }


# ==========================================
# HOTEL AGENT
# ==========================================

def hotel_agent(state: TravelState):

    print("\n[Hotel Agent] Searching for hotels...")

    query = f"Best hotels for {state['user_query']}"

    hotel_results = tavily_search(query)

    return {

        "hotel_results": hotel_results,

        "messages": [
            AIMessage(
                content="Hotel information retrieved."
            )
        ]

    }


# ==========================================
# ITINERARY AGENT
# ==========================================

def itinerary_agent(state: TravelState):

    print("\n[Itinerary Agent] Creating travel itinerary...")

    prompt = f"""

You are an expert travel planner.

Create a detailed and practical travel itinerary.

USER REQUEST:
{state['user_query']}


FLIGHT INFORMATION:
{state['flight_results']}


HOTEL INFORMATION:
{state['hotel_results']}


INSTRUCTIONS:

1. Create a day-by-day travel itinerary.
2. Consider the user's total budget.
3. Include suitable tourist attractions.
4. Suggest practical transportation options.
5. Recommend restaurants and food options.
6. Consider travel time between attractions.
7. Avoid unrealistic schedules.
8. Provide estimated expenses.
9. Clearly distinguish verified information from estimates.
10. Never invent confirmed flight or hotel bookings.

If information is unavailable, clearly mention it.

Make the itinerary easy to understand and follow.

"""

    response = llm.invoke([

        SystemMessage(
            content="You are an expert AI travel planning assistant."
        ),

        HumanMessage(
            content=prompt
        )

    ])

    return {

        "itinerary": response.content,

        "messages": [
            response
        ],

        "llm_calls": state.get("llm_calls", 0) + 1

    }


# ==========================================
# FINAL RESPONSE AGENT
# ==========================================

def final_agent(state: TravelState):

    print("\n[Final Agent] Preparing final travel plan...")

    final_prompt = f"""

You are a professional AI travel planning assistant.

Prepare the final travel plan using the information below.

USER REQUEST:
{state['user_query']}


FLIGHT RESULTS:
{state['flight_results']}


HOTEL RESULTS:
{state['hotel_results']}


GENERATED ITINERARY:
{state['itinerary']}


FORMAT YOUR RESPONSE USING THESE SECTIONS:


1. TRIP SUMMARY

- Destination
- Duration
- Starting location
- Total budget


2. FLIGHT INFORMATION

- Available flight information
- Departure and arrival details if available
- Estimated prices if available

Mention if live flight prices are unavailable.


3. HOTEL SUGGESTIONS

- Hotel name
- Location
- Available information
- Estimated price if available

Do not claim that a hotel is booked.


4. DAY-BY-DAY ITINERARY

Provide a clear schedule for each day.


5. ESTIMATED BUDGET

Break down expenses into:

- Flights
- Accommodation
- Food
- Transportation
- Sightseeing
- Miscellaneous

Show the estimated total.

Ensure the estimated expenses do not exceed the user's budget.
If the budget cannot realistically cover the trip, explain why.


6. FINAL RECOMMENDATIONS

Provide useful travel tips and recommendations.


IMPORTANT:

- Be practical and realistic.
- Do not fabricate booking confirmations.
- Do not present estimated prices as verified prices.
- Mention information that requires verification.
- Clearly identify missing information.
- Keep the response organized and readable.

"""

    response = llm.invoke([

        SystemMessage(
            content="You are a professional AI travel planning assistant."
        ),

        HumanMessage(
            content=final_prompt
        )

    ])

    return {

        "messages": [
            response
        ],

        "llm_calls": state.get("llm_calls", 0) + 1

    }


# ==========================================
# BUILD LANGGRAPH WORKFLOW
# ==========================================

graph = StateGraph(TravelState)


# Register agents

graph.add_node(
    "flight_agent",
    flight_agent
)

graph.add_node(
    "hotel_agent",
    hotel_agent
)

graph.add_node(
    "itinerary_agent",
    itinerary_agent
)

graph.add_node(
    "final_agent",
    final_agent
)


# Define execution sequence

graph.add_edge(
    START,
    "flight_agent"
)

graph.add_edge(
    "flight_agent",
    "hotel_agent"
)

graph.add_edge(
    "hotel_agent",
    "itinerary_agent"
)

graph.add_edge(
    "itinerary_agent",
    "final_agent"
)

graph.add_edge(
    "final_agent",
    END
)


# ==========================================
# POSTGRESQL CHECKPOINTER
# ==========================================

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)

# Initialize checkpoint tables

checkpointer.setup()


# Compile LangGraph

travel_graph = graph.compile(
    checkpointer=checkpointer
)


# ==========================================
# MAIN FUNCTION FOR FASTAPI
# ==========================================

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None
):

    # Generate unique thread ID if not provided

    if not thread_id:

        thread_id = f"user_{uuid.uuid4().hex}"


    # LangGraph configuration

    config = {

        "configurable": {

            "thread_id": thread_id

        }

    }


    # Initial state

    initial_state = {

        "messages": [

            HumanMessage(
                content=user_input
            )

        ],

        "user_query": user_input,

        "flight_results": "",

        "hotel_results": "",

        "itinerary": "",

        "llm_calls": 0

    }


    # Execute travel planning workflow

    result = travel_graph.invoke(
        initial_state,
        config=config
    )


    # Extract final response

    final_answer = result["messages"][-1].content


    # Return structured response

    return {

        "thread_id": thread_id,

        "answer": final_answer,

        "flight_results": result.get(
            "flight_results", ""
        ),

        "hotel_results": result.get(
            "hotel_results", ""
        ),

        "itinerary": result.get(
            "itinerary", ""
        ),

        "llm_calls": result.get(
            "llm_calls", 0
        )

    }


# ==========================================
# END OF BACKEND
# ==========================================