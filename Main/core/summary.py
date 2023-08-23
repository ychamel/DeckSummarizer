import openai

from Main.core.qa import query_folder
import streamlit as st


def complete(prompt):
    messages = [
        {"role": "system",
         "content": "You are an excelent analyst that writes report based on a given topic and the information supplied regarding it."
         },
    ]
    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages
    )
    answer = ""
    for choice in response.choices:
        answer += choice.message.content
    return answer


def write_analysis(topic, folder_index):
    # retrieve relevant data
    query = f"get me all information regarding the following topic: {topic}"
    retrieved = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=True,
        model="openai",
        openai_api_key=st.session_state.get("OPENAI_API_KEY"),
        temperature=0,
    )
    prompt = f"Write a detailed analysis on the following topic: {topic}, based on the info bellow: \n" \
             f"{retrieved}"
    # prompt chatgpt for result
    result = complete(prompt)
    return result


def write_report(folder_index):
    topics = [
        "Company Overview",
        "Company Headcount",
        "Number of Clients",
        "Geography Presence",
        "Number of Products",
        "Key Milestones and Figures",
        "Market Analysis",
        "Products/Services Offering",
        "Business Model",
        "Pricing",
        "Financial Analysis (Tables at the end + Graphs)",
        "Strategy Analysis",
        "Final Recommendations and Analysis",
    ]
    out = ""

    for topic in topics:
        out += f"\n \n{topic}: \n \n"
        out += write_analysis(topic, folder_index)

    return out
