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
        model="gpt-3.5-turbo",
        messages=messages
    )
    answer = ""
    for choice in response.choices:
        answer += choice.message.content
    return answer


def write_analysis(title, topic, folder_index):
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
    prompt = f"Write a detailed report on the following topic: {title}, based on the info below. Don't refer to the sources given in the retrieved data, and if there is not enough data just say that not enough data was supplied. \n" \
             f"info: \n" \
             f"{retrieved}"
    # prompt chatgpt for result
    result = complete(prompt)
    return result


def write_report(folder_index):
    topics = {
        "Company Overview": "Company Overview, this includes Company Headcount, Number of Clients, Geography Presence, Number of Products, and Key Milestones and Figures ",
        "Market Analysis": "the market analysis for the company and a detailed assessment of the business's target market and the competitive landscape within their specific industry",
        "Products/Services Offering": "the product or service offering being sold by the company",
        "Business Model": "The buisness model of the company",
        "Pricing": "The pricing of the company and their products",
        "Financial Analysis": "The Financials of the company including the balance sheet, the income statement, and the cash flow statement",
        "Strategy Analysis": "The strategy of the company and how they plan to approach the market",
        "Final Recommendations and Analysis": "the company's market approach and their financials",
    }
    out = ""

    for title, topic in topics.items():
        out += f"\n \n{title}: \n \n"
        out += write_analysis(title, topic, folder_index)

    return out
