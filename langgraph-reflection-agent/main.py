from typing import List, Sequence
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflect_chain

load_dotenv()


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflect_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": state})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflect_node)

builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)


graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = HumanMessage(
        content="""Make this tweet better:"""
        """o unico risco que vejo na IA é eu não ficar milionario com ela ou com GPU"""
    )

    response = graph.invoke(inputs)
