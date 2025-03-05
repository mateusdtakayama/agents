from typing import Literal
from dotenv import load_dotenv
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent

import random

from langchain_openai.chat_models import ChatOpenAI


load_dotenv()


@tool
def triple(num: float) -> float:
    """
    :param num: a number to triple
    :return: the number tripled -> multiplied by 3
    """

    return 3 * float(num)


@tool
def random_number() -> float:
    """
    :return: a random number
    """
    return random.random()


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [triple, random_number, get_weather]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


graph = create_react_agent(llm, tools=tools)
