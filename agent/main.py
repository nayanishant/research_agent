import os
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from pydantic import BaseModel, Field
import requests
from operator import add
from crawl4ai import AsyncWebCrawler
import asyncio
import subprocess


load_dotenv()

# LLM Setup
flash_2_5_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

ollama_gemma_llm = ChatOllama(
    model="llama3.2",
    base_url="http://192.168.1.102:11434",
    temperature=0.3,
    stream=True,
)

openai_gpt_5_mini = ChatOpenAI(
    model="gpt-5-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    streaming=True,
)


# Schema for structured output for user's query
class SearchQuery(BaseModel):
    refactored_query: List[str] = Field(
        None,
        description="The list of refactored queries that will help perform web search from user's query.",
    )


# State
class State(TypedDict):
    user_query: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    search_queries: list[str] | None
    web_search_results: list[any] | None
    urls: list[str] | None
    scraped_data: Annotated[list[dict], add]
    summary: str


def start_workflow(state: State):
    """Start Of Research Workflow"""
    user_input = state["messages"][-1].content
    return {"user_query": user_input}


# Function to generate 2-3 query from the user's query
def refactor_result(state: State):
    """Create 3 to 4 refactored queries from user's query."""
    user_input = state["user_query"]
    structured_response = flash_2_5_llm.with_structured_output(SearchQuery)
    final_response = structured_response.invoke(
        [
            SystemMessage(
                content="You are an user query analyser. Your job is to analyse user query and generate 2-3 search engine optimised queries based on what user asked."
            ),
            HumanMessage(content=user_input),
        ]
    )

    # print(final_response.model_dump())

    return {"search_queries": final_response.refactored_query}


# Function to search the web based on refactored_query
def run_search(state: State):
    """Performs web searches for each refactored query in the state and aggregates results. Extracts unique URLs from the search responses and returns them as a state update."""
    print(len(state["search_queries"]))
    results: list[any] = []
    urls: list[str] = []
    for query in state["search_queries"]:
        print(query)
        response = requests.get(f"http://localhost:8080/search?format=json&q={query}")
        json_data = response.json()
        results.extend(json_data["results"][:2])
    # print(len(results))

    for url in results:
        urls.append(url["url"])

    return {"web_search_results": results, "urls": list(set(urls))}


# Function to send each url to scrape it's content
def route_to_scrape(state: State):
    return [Send("scrape_data", {"url": url}) for url in state["urls"]]


# Function to scrape the data of each URL
def scrape_data(state: dict):
    url = state["url"]

    try:
        crawler = AsyncWebCrawler()
        result = asyncio.run(
            crawler.arun(
                url=url,
                extraction_strategy="readability",
            )
        )
        crawler.close()

        content = (
            getattr(result, "extracted_content", None)
            or getattr(result, "main_content", None)
            or getattr(result, "markdown", None)
        )

        if not content:
            raise ValueError("No content extracted")

        content = content[:4000]

        return {
            "scraped_data": [
                {
                    "content": content,
                    "status": "success",
                }
            ]
        }

    except Exception as e:
        return {
            "scraped_data": [
                {
                    "url": url,
                    "error": str(e),
                    "status": "failed",
                }
            ]
        }


# Function to merge the scraped data of each URL
def merge_scraped_data(state: State) -> dict:
    successful = [
        item for item in state["scraped_data"] if item.get("status") == "success"
    ]
    failed = len(state["scraped_data"]) - len(successful)
    # print(f"Scraping complete: {len(successful)} success, {failed} failed")
    subprocess.run(["killall", "chrome"])
    return {}


# Function to summarize the scraped data
def summarize_scraped_data(state: State):
    """Summarize successfully scraped web content.

    Aggregates all successful scraped results and generates a concise, research-oriented summary.

    Returns a partial state update with the final summary.
    """

    scraped_items = state.get("scraped_data", [])

    # Collect only successful content
    contents: list[str] = [
        item["content"]
        for item in scraped_items
        if item.get("status") == "success" and item.get("content")
    ]

    if not contents:
        return {"summary": "No usable content was scraped."}

    combined_text = "\n\n---\n\n".join(contents)
    combined_text = f"User Query: {state['user_query']} \n\nContext: {combined_text}"

    response = openai_gpt_5_mini.invoke(
        [
            SystemMessage(
                content=(
                    "You are an expert research assistant.\n\n"
                    "Your task is to explain the user's query clearly and accurately using only the provided content.\n\n"
                    "STRICT RULES:\n"
                    "- Use ONLY the provided content.\n"
                    "- Do NOT add outside knowledge.\n"
                    "- Do NOT hallucinate or guess.\n"
                    "- If information is incomplete or unclear, say so explicitly.\n\n"
                    "OUTPUT STRUCTURE:\n"
                    "1. Start with a short, direct answer (2–3 sentences).\n"
                    "2. Then organize the explanation into clear sections based on the topic.\n"
                    "   Examples of section titles (adapt as needed):\n"
                    "   - Overview\n"
                    "   - How it works / How it started\n"
                    "   - Key details / Developments\n"
                    "   - Benefits / Impact\n"
                    "   - Limitations / Risks\n"
                    "FORMATTING:\n"
                    "- Use plain text only (no markdown symbols like #, *, or backticks).\n"
                    "- Use short paragraphs or bullet points.\n"
                    "- Keep spacing clean and readable.\n"
                    "- Avoid long dense blocks of text.\n\n"
                    "TONE:\n"
                    "- Neutral, clear, and professional\n"
                    "- Easy to understand\n"
                    "- Avoid jargon unless necessary (and explain it if used)\n\n"
                    "CONTENT GUIDELINES:\n"
                    "- Focus on the most important and relevant information\n"
                    "- Avoid repetition\n"
                    "- Prefer clarity over completeness\n"
                    "- Include practical implications if relevant to the query\n\n"
                    "DO NOT:\n"
                    "- Do not include citations like [1], [2]\n"
                    "- Do not mention 'based on the provided content'\n"
                    "- Do not add unnecessary introductions or conclusions\n"
                )
            ),
            HumanMessage(content=f"{combined_text}"),
        ]
    )
    # print(response.content)
    return {"summary": response.content}


graph = StateGraph(State)

# Define nodes
graph.add_node("start_workflow", start_workflow)
graph.add_node("refactor_result", refactor_result)
graph.add_node("run_search", run_search)
graph.add_node("route_to_scrape", route_to_scrape)
graph.add_node("scrape_data", scrape_data)
graph.add_node("merge_scraped_data", merge_scraped_data)
graph.add_node("summarize_scraped_data", summarize_scraped_data)

# Define edge
graph.add_edge(START, "start_workflow")
graph.add_edge("start_workflow", "refactor_result")
graph.add_edge("refactor_result", "run_search")
graph.add_conditional_edges("run_search", route_to_scrape, None)

graph.add_edge("scrape_data", "merge_scraped_data")
graph.add_edge("merge_scraped_data", "summarize_scraped_data")
graph.add_edge("summarize_scraped_data", END)

app = graph.compile()

# print(app.get_graph().draw_mermaid())


# async def main():
#     user_input = input("Please enter your query: ")
#     result = await app.ainvoke({"messages": [HumanMessage(content=user_input)]})
#     print("Summary:", result["summary"])


# asyncio.run(main())
