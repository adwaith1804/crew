import streamlit as st
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from datetime import datetime, timedelta

st.set_page_config(page_title="AI Trip Planner", page_icon="✈️", layout="wide")

if "trip_plan" not in st.session_state:
    st.session_state["trip_plan"] = None


def create_agents(api_key: str):
    llm = ChatGroq(
        model="llama3-70b-8192",   # or "mixtral-8x7b-32768" / "gemma2-9b-it"
        temperature=0.7,
        groq_api_key=api_key
    )

    city_expert = Agent(
        role="City Information Expert",
        goal="Provide detailed information about the city, including attractions, culture, and local tips.",
        backstory=(
            "Experienced travel researcher with extensive knowledge of global destinations. "
            "Passionate about helping travelers discover hidden gems and plan unforgettable trips."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    itinerary_planner = Agent(
        role="Itinerary Planner",
        goal="Create a detailed and personalized itinerary based on the user's preferences and the information provided by the city expert.",
        backstory=(
            "Skilled travel planner with a knack for crafting unique and memorable travel experiences. "
            "Dedicated to designing itineraries that cater to individual interests and needs."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    return city_expert, itinerary_planner


def create_tasks(city_expert, itinerary_planner, origin, destination,
                 start_date, end_date, interests, budget, travel_style):

    duration = (end_date - start_date).days + 1
    interests_str = ", ".join(interests) if interests else "general sightseeing"

    research_task = Task(
        description=f"""Research {destination} and provide:
        - Top 5 must-visit attractions related to {interests_str}
        - Best restaurants, local events, and cultural insights
        - Cultural highlights and local tips for travelers
        - Transportation options and estimated costs
        - Weather conditions during the travel dates ({start_date} to {end_date})
        - Safety tips and health advisories
        - Focus on {budget} budget and {travel_style} travel style. Be concise and informative.
        """,
        agent=city_expert,
        expected_output="Structured city info with practical travel tips and recommendations."
    )

    itinerary_task = Task(
        description=f"""Create a {duration}-day itinerary for a trip from {origin} to {destination}
        based on the following preferences:
        Dates: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
        Interests: {interests_str} | Budget: {budget} | Travel Style: {travel_style}

        For each day, include:
        - Morning, Afternoon, and Evening activities related to {interests_str}
        - Recommended restaurants and local events
        - Cultural experiences and local tips
        - Transportation options and estimated costs
        - Weather considerations for each day
        - Safety tips and health advisories
        - Focus on {budget} budget and {travel_style} travel style. Be concise and practical.
        """,
        agent=itinerary_planner,
        expected_output="Structured daily itinerary with activities, dining, and cultural experiences.",
        context=[research_task]
    )

    return [research_task, itinerary_task]


def generate_trip_plan(api_key, origin, destination, start_date, end_date,
                       interests, budget, travel_style):
    try:
        city_expert, itinerary_planner = create_agents(api_key)
        tasks = create_tasks(
            city_expert, itinerary_planner,
            origin, destination,
            start_date, end_date,
            interests, budget, travel_style
        )
        crew = Crew(
            agents=[city_expert, itinerary_planner],
            tasks=tasks,
            verbose=True
        )
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"An error occurred: {str(e)}. Please check your API key and try again."


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password",
                            help="Get your free key at https://console.groq.com")

    st.markdown("**Recommended models:**")
    st.code("llama3-70b-8192\nmixtral-8x7b-32768\ngemma2-9b-it")
    st.markdown("---")
    st.markdown("Built with [CrewAI](https://crewai.com) + [Groq](https://groq.com)")

# ── Main UI ──────────────────────────────────────────────────────────────────
st.title("✈️ AI Trip Planner")
st.markdown("Plan your perfect trip powered by **Groq** (free & blazing fast).")

with st.form("trip_planner_form"):
    col1, col2 = st.columns(2)

    with col1:
        origin = st.text_input("Origin City", placeholder="e.g. Mumbai")
        destination = st.text_input("Destination City", placeholder="e.g. Paris")

    with col2:
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")

    interests = st.multiselect(
        "Interests",
        ["Culture", "Nature", "Food", "History", "Adventure", "Relaxation"],
        default=["Culture", "Food"]
    )
    col3, col4 = st.columns(2)
    with col3:
        budget = st.selectbox("Budget", ["Low", "Medium", "High"])
    with col4:
        travel_style = st.selectbox("Travel Style", ["Solo", "Couple", "Family", "Group"])

    submit_button = st.form_submit_button(label="🗺️ Generate Trip Plan", use_container_width=True)

# ── Handle submission ─────────────────────────────────────────────────────────
if submit_button:
    # Validation
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif not origin or not destination:
        st.error("Please enter both origin and destination cities.")
    elif end_date <= start_date:
        st.error("End date must be after the start date.")
    else:
        with st.spinner("🤖 AI agents are planning your trip... This may take a minute."):
            plan = generate_trip_plan(
                api_key, origin, destination,
                start_date, end_date,
                interests, budget, travel_style
            )
            st.session_state["trip_plan"] = plan

# ── Display result ────────────────────────────────────────────────────────────
if st.session_state["trip_plan"]:
    st.markdown("---")
    st.subheader(f"🗺️ Your Trip Plan: {origin} → {destination}")
    st.markdown(st.session_state["trip_plan"])

    st.download_button(
        label="📥 Download Trip Plan",
        data=st.session_state["trip_plan"],
        file_name=f"trip_plan_{destination.replace(' ', '_')}.txt",
        mime="text/plain"
    )