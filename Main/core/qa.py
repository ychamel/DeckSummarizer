from typing import Any, List

import openai
import streamlit as st
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from Main.core.prompts import STUFF_PROMPT
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from Main.core.embedding import FolderIndex
from Main.core.debug import FakeChatModel
from pydantic import BaseModel
from wolframalpha import Client


class AnswerWithSources(BaseModel):
    answer: str
    sources: List[Document]


def text_analyse(text: str):
    substrings = []
    start_index = text.find("{")

    while start_index != -1:
        end_index = text.find("}", start_index + 1)
        if end_index != -1:
            substrings.append([start_index + 1, end_index])
        start_index = text.find("{", start_index + 1)

    for substring in reversed(substrings):
        start = substring[0]
        end = substring[1]
        string = text[start:end]
        try:
            out = Wolf_analyse(string)
        except Exception as e:
            print(f"error: {e}")
            out = string
        text = text[:start] + out + text[end:]
    return text


def Wolf_analyse(text: str):
    Wolf_client = Client(st.session_state.get("WOLFRAMALPHA_KEY", ""))

    res = Wolf_client.query(text)

    answer = next(res.results).text

    return text+":"+answer


def get_relevant_docs(query: str, search_query: str, folder_index: FolderIndex) -> AnswerWithSources:
    relevant_docs = folder_index.index.similarity_search(search_query)

    messages = [
        {"role": "system",
         "content": "Create a final answer to the given questions using the provided document excerpts(in no particular order) as references. \n"
                    "for calculations return the equation between brackets which will be passed to a seperate api to do the calculation. \n"
                    "ex: info: 'curr period val=1000, last period val=2000' question: 'what's the year on year growth?' answer: '{(1000-2000)/2000*100}'"
                    f"The context is the following: {relevant_docs}"
         },
        {"role": "user", "content": f"question: {query}"}
    ]
    # f"some key topics to cover are {topics.keys()} described as follows {topics}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        temperature=0,
    )
    answer = ""
    for choice in response.choices:
        answer += choice.message.content

    answer = text_analyse(answer)
    return AnswerWithSources(answer=answer, sources=relevant_docs)


def query_folder(
        query: str,
        folder_index: FolderIndex,
        return_all: bool = False,
        model: str = "openai",
        **model_kwargs: Any,
) -> AnswerWithSources:
    """Queries a folder index for an answer.

    Args:
        query (str): The query to search for.
        folder_index (FolderIndex): The folder index to search.
        return_all (bool): Whether to return all the documents from the embedding or
        just the sources for the answer.
        model (str): The model to use for the answer generation.
        **model_kwargs (Any): Keyword arguments for the model.

    Returns:
        AnswerWithSources: The answer and the source documents.
    """
    supported_models = {
        "openai": ChatOpenAI,
        "debug": FakeChatModel,
    }

    if model in supported_models:
        llm = supported_models[model](**model_kwargs)
    else:
        raise ValueError(f"Model {model} not supported.")

    chain = load_qa_with_sources_chain(
        llm=llm,
        chain_type="stuff",
        prompt=STUFF_PROMPT,
    )

    relevant_docs = folder_index.index.similarity_search(query)
    result = chain(
        {"input_documents": relevant_docs, "question": query}, return_only_outputs=True
    )
    sources = relevant_docs

    if not return_all:
        sources = get_sources(result["output_text"], folder_index)

    answer = result["output_text"].split("SOURCES: ")[0]
    answer = text_analyse(answer)
    return AnswerWithSources(answer=answer, sources=sources)


def get_sources(answer: str, folder_index: FolderIndex) -> List[Document]:
    """Retrieves the docs that were used to answer the question the generated answer."""

    source_keys = [s for s in answer.split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for file in folder_index.files:
        for doc in file.docs:
            if doc.metadata["source"] in source_keys:
                source_docs.append(doc)

    return source_docs


def get_query_answer(query, summary):
    messages = [
        {"role": "system",
         "content": "You are an answer generator for a search engine, you will be given a question and you'll return a list of relevant keywords to look for. \n"
                    "ex: Q: 'what is the net operational profit in 2022?', A: 'Buisness data, gross profit, operating expenses, net sales, revenue, cost of sales, etc.' "
                    f"Context: {summary}"
         },
        {"role": "user", "content": f"question: {query}"}
    ]
    # f"some key topics to cover are {topics.keys()} described as follows {topics}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    answer = ""
    for choice in response.choices:
        answer += choice.message.content
    return answer
