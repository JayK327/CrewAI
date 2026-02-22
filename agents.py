from crewai import Agent
from tools import yt_tool
import os
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", streaming=True)


# Create a senior blog content researcher
blog_researcher = Agent(
    role="Senior Blog Content Researcher",
    goal='Get relevant video content for the topic {topic} from the YouTube channel.',
    backstory=(
        "Expert in understanding videos in dpeth with respect to Machone Learning, Deep Learning and GenAI models"
    ),
    tools=[yt_tool],
    llm=llm,
    verbose=True,
    memory=False,
    allow_delegation=False
)

# Create a senior blog writer agent with YouTube tool
blog_writer = Agent(
    role="Senior Blog Writer",
    goal='Narrate compelling tech stories about the videop {topic} from YT channel',
    backstory=(
        "You are an experienced blog writer who turns complex research into "
        "compelling stories. You understand content structure, search intent, "
        "and how to incorporate insights from articles and YouTube videos naturally."
    ),
    tools=[yt_tool],
    llm=llm,
    verbose=True,
    memory=False,
    allow_delegation=False
)
