import os
import pinecone
import openai
import tiktoken

from langchain.text_splitter import RecursiveCharacterTextSplitter

from Main.core.parsing import File
from Main.core.qa import query_folder
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


def write_analysis(title, topic, folder_index):
    # retrieve relevant data
    query = f"get me all information regarding the following topic: {topic}"
    retrieved = retrieve(query, folder_index)
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
