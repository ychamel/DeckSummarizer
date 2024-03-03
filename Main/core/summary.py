import json
import os
import pinecone
import openai
import tiktoken

from langchain.text_splitter import RecursiveCharacterTextSplitter

from Main.core.parsing import File
from Main.core.qa import query_folder, get_relevant_docs
import streamlit as st


# create the length function
def tiktoken_len(text: str):
    tokens = tiktoken.get_encoding('p50k_base').encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


# retrieve files from pinecone
def retrieve(query: str, index):
    """
    retrieve data from the db using the query
    :param query:
    :return:
    """
    res = openai.Embedding.create(
        input=[query],
        engine="text-embedding-ada-002"
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]
    prompt = None
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= 3500:
            prompt = (
                "\n\n---\n\n".join(contexts[:i - 1])
            )
            break
        elif i == len(contexts) - 1:
            prompt = (
                "\n\n---\n\n".join(contexts)
            )
    return prompt


def store_txt(files: list[File]):
    input_txt = ""
    for file in files:
        for doc in file.docs:
            input_txt += doc.page_content + '\n'

    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,  # number of tokens overlap between chunks
        length_function=tiktoken_len,
        separators=['\n\n', '\n', ' ', '']
    )
    chunks = text_splitter.split_text(input_txt)
    # create Embedding
    embed_model = "text-embedding-ada-002"
    res = openai.Embedding.create(
        input=chunks
        , engine=embed_model
    )
    data = [
        {
            'id': f'{i}',
            'text': chunk,
        } for i, chunk in enumerate(chunks)
    ]
    index = init_pinecone(res['data'][0]['embedding'])
    store_data(data, embed_model, index)
    return index


def init_pinecone(dimension):
    pinecone.init(api_key=st.session_state.get('PINECONE_API_KEY'),
                  enviroment=st.session_state.get('PINECONE_ENVIRONMENT'))

    # check if index already exists (it shouldn't if this is first time)
    if st.session_state.get("PINECONE_INDEX") not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            st.session_state.get("PINECONE_INDEX"),
            dimension=len(dimension),
            metric='cosine',
            metadata_config={'indexed': ['channel_id', 'published']}
        )
    # connect to index
    index = pinecone.Index(st.session_state.get("PINECONE_INDEX"))
    # clear old data
    index.delete(deleteAll='true')
    # view index stats
    # self.index.describe_index_stats()
    return index


def store_data(data, embed_model, index):
    from tqdm.auto import tqdm
    from time import sleep

    batch_size = 100  # how many embeddings we create and insert at onceq
    for i in tqdm(range(0, len(data), batch_size)):
        # find end of batch
        i_end = min(len(data), i + batch_size)
        meta_batch = data[i:i_end]
        # get ids
        ids_batch = [x['id'] for x in meta_batch]
        # get texts to encode
        texts = [x['text'] for x in meta_batch]
        # create embeddings (try-except added to avoid RateLimitError)
        try:
            res = openai.Embedding.create(input=texts, engine=embed_model)
        except:
            done = False
            while not done:
                sleep(5)
                try:
                    res = openai.Embedding.create(input=texts, engine=embed_model)
                    done = True
                except:
                    pass
        embeds = [record['embedding'] for record in res['data']]
        # cleanup metadata
        meta_batch = [{
            'text': x['text'],
        } for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        # upsert to Pinecone
        index.upsert(vectors=to_upsert)


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


def write_analysis(topic, folder_index):
    # retrieve relevant data
    query = f"{topic}"
    retrieved = retrieve(query, folder_index)
    prompt = f"Write a short essay on {topic}, given the following information: \n" \
             f"{retrieved}"
    # prompt chatgpt for result
    result = complete(prompt)
    return result


def write_report(folder_index):
    topics = {
        "Company Overview": "The Company Overview, this includes Company Headcount, Number of Clients, Geography Presence, Number of Products, and Key Milestones and Figures ",
        "Market Analysis": "The market analysis for the company and a detailed assessment of the business's target market and the competitive landscape within their specific industry",
        "Products/Services Offering": "the products and/or services offering being sold by the company",
        "Business Model": "The buisness model of the company",
        "Pricing": "The company's pricing",
        "Financial Analysis": "The Financial analysis of the company including the balance sheet, the income statement, and the cash flow statement and any given financial reports.",
        "Strategy Analysis": "The strategy analysis of the company and how they plan to approach the market",
        "Final Recommendations and Analysis": "final recomendation and analysis for the company's market approach and their financials",
    }
    out = ""

    for title, topic in topics.items():
        out += f"\n \n{title}: \n \n"
        out += write_analysis(topic, folder_index)

    return out


def write_RSM(files: list[File]):
    topics = {
        "Company Overview": "The Company Overview, this includes Company Headcount, Number of Clients, Geography Presence, Number of Products, and Key Milestones and Figures ",
        "Market Analysis": "The market analysis for the company and a detailed assessment of the business's target market and the competitive landscape within their specific industry",
        "Products/Services Offering": "the products and/or services offering being sold by the company",
        "Business Model": "The buisness model of the company",
        "Pricing": "The company's pricing",
        "Financial Analysis": "The Financial analysis of the company including the balance sheet, the income statement, and the cash flow statement and any given financial reports.",
        "Strategy Analysis": "The strategy analysis of the company and how they plan to approach the market",
        "Final Recommendations and Analysis": "final recomendation and analysis for the company's market approach and their financials",
    }
    topics2 = [
        "Company Overview", "Business Overview", "Transaction Overview / Structure", "Investment Highlights",
        "Investment Considerations",
    ]
    # get input text
    input_txt = ""
    for file in files:
        for doc in file.docs:
            input_txt += doc.page_content + '\n'
    messages = [
        {"role": "system",
         "content": "You are an excelent analyst that will be given a CIM document of a comapny, and you are tasked to write an RSM report that includes a comprehensive analysis of the financial health and potential of the company. \n"
                    f"some key topics to be covered are the following {topics2}. \n"
                    f"Go into as much detail as you can in each key topics if the data exists in the CIM."
         },
        {"role": "user", "content": f"here is the CIM document: \n {input_txt}"}
    ]
    # f"some key topics to cover are {topics.keys()} described as follows {topics}."
    response = openai.ChatCompletion.create(
        model="gpt-4-0125-preview",
        messages=messages
    )
    answer = ""
    for choice in response.choices:
        answer += choice.message.content
    return answer


def get_summary(file: File):
    # get input text
    input_txt = ""
    for doc in file.docs:
        input_txt += doc.page_content + '\n'
    messages = [
        {"role": "system",
         "content": "You are a text summariser that take a chunk of text and returns its summary."
         },
        {"role": "user", "content": f"text: {input_txt}"}
    ]
    # f"some key topics to cover are {topics.keys()} described as follows {topics}."
    response = openai.ChatCompletion.create(
        model="gpt-4-0125-preview",
        messages=messages
    )
    answer = ""
    for choice in response.choices:
        answer += choice.message.content
    return answer


def website_summary(folder_index):
    # for each keyword fetch data
    Topics = ["company profile, mission, values", "Products and Services", "Management Team",
              "Financial Reports and Investor Relations", "Clientele and Partenerships",
              "News and Press Release", "Contact Information", "Legal and Regulatory Compliance", "Social Media Links",
              "Client Testimonials", "Awards and Accolades", "Industry Affiliations", "CSR initiatives", "Job Openings",
              "Events and Conferences"]
    # filter redundant data
    Docs = {}
    for topic in Topics:
        relevant_docs = folder_index.index.similarity_search(topic, k=3)
        for doc in relevant_docs:
            id = doc.metadata.get("file_id") + ":" + doc.metadata.get("source")
            if id not in Docs:
                Docs[id] = doc

    # given all the data generate a summary
    # get input text
    input_txt = ""
    for doc in Docs.values():
        input_txt += doc.page_content + '\n'
    messages = [
        {"role": "system",
         "content": f"You are a text summariser that takes a website parsed data and return a detailed summary covering the following topics {Topics}."
         },
        {"role": "user", "content": input_txt}
    ]
    # f"some key topics to cover are {topics.keys()} described as follows {topics}."
    response = openai.ChatCompletion.create(
        model="gpt-4-0125-preview",
        messages=messages,
        max_tokens=4000

    )
    answer = ""
    for choice in response.choices:
        answer += choice.message.content
    return answer


def get_ratio_analysis(folder_index):
    data_sheet = {
        "Income Statement": {
            "Revenue": 0,
            "Cost of Goods Sold": 0,
            "Interest Expense": 0,
            "Tax Expense": 0,
            "Income from Cont Operations": 0,
            "Net Income": 0
        },
        "Balance Sheet": {
            "Cash": 0,
            "Short Term Investments": 0,
            "Accounts Receivable": 0,
            "Inventory": 0,
            "Current Assets": 0,
            "Long Term Investments": 0,
            "Net Fixed Assets": 0,
            "Other Assets": 0,
            "Total Assets": 0,
            "Current Liabilities": 0,
            "Total Liabilities": 0,
            "Total Stockholders' Equity": 0,
        },
        "Cash Flow": {
            "Cash Flow from Operations": 0,
            "Dividends Paid": 0,
            "Interest Paid": 0,
        },
        "Share Information ": {
            "Market Price at Year End": 0,
            "Earnings Per Share - Basic": 0,
            "Shares Outstanding": 0,
        }
    }
    ratio_sheet = {
        "Growth Ratios": {
            "Sales Growth": 0,
            "Income Growth": 0,
            "Asset Growth": 0,
        },
        "Profitability Ratios": {
            "Profit Margin": 0,
            "Return on Assets": 0,
            "Return on Equity": 0,
            "Dividend Payout Ratio": 0,
            "Price Earnings Ratio": 0,
        },
        "Activity Ratios": {
            "Receivable Turnover": 0,
            "Inventory Turnover": 0,
            "Fixed Asset Turnover": 0,
        },
        "Liquidity Ratios": {
            "Current Ratio": 0,
            "Quick Ratio": 0,
        },
        "Solvency Ratios": {
            "Debt to Total Assets": 0
        },
        "Year": ""
    }
    ratio_sheet_example = {
        "Growth Ratios": {
            "Sales Growth": 10.5,
            "Income Growth": -15.0,
            "Asset Growth": 3.5,
        },
        "Profitability Ratios": {
            "Profit Margin": 1.1,
            "Return on Assets": 2.1,
            "Return on Equity": 0,
            "Dividend Payout Ratio": 1.0,
            "Price Earnings Ratio": 5.2,
        },
        "Activity Ratios": {
            "Receivable Turnover": 1.0,
            "Inventory Turnover": 0.5,
            "Fixed Asset Turnover": 2.1,
        },
        "Liquidity Ratios": {
            "Current Ratio": 3.0,
            "Quick Ratio": 0.5,
        },
        "Solvency Ratios": {
            "Debt to Total Assets": 10.2
        },
        "Year": "2022",
        "Notes": ""
    }
    # extract data related to these ratios
    Docs = {}
    for id, val in data_sheet.items():
        topic = f"{id} : {val.keys()}"
        relevant_docs = folder_index.index.similarity_search(topic, k=4)
        for doc in relevant_docs:
            id = doc.metadata.get("file_id") + ":" + doc.metadata.get("source")
            if id not in Docs:
                Docs[id] = doc
    # fill values
    # get input text
    input_txt = ""
    for doc in Docs.values():
        input_txt += doc.page_content + '\n'
    messages = [
        {"role": "system",
         "content": f"You are a professional accountant that's tasked to do a Ratio Analysis using the given text and return it in the following json format: {json.dumps(ratio_sheet)}. \n"
                    f"Don't add any extra text except the json formated output, since it's expected to be parsed as json later. \n"
                    f"example output: \n {json.dumps(ratio_sheet)}"
         },
        {"role": "user", "content": input_txt}
    ]
    # f"some key topics to cover are {topics.keys()} described as follows {topics}."
    response = openai.ChatCompletion.create(
        model="gpt-4-0125-preview",
        messages=messages,
        max_tokens=4000,
        temperature=0,

    )
    answer = ""
    for choice in response.choices:
        answer += choice.message.content
    return json.loads(answer)
    # generate ratios
