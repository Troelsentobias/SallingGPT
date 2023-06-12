"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import pandas as pd


# Here we load in the data in the format that Notion exports it in.
# ps = list(Path("Notion_DB/").glob("**/*.md"))
ps = pd.read_excel("Interflora/Interflora - FAQ.xlsx")

data = []
sources = []
for index, row in ps.iterrows():
    data.append(row["Svar"])
    sources.append(row["Spørgsmål"])

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

openai_api_key = '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'

# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_key='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'), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
