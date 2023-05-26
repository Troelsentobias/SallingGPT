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
ps = pd.read_excel("SG/support_bilka.xlsx")

data = []
sources = []
for index, row in ps.iterrows():
    data.append(row["Article Body"])
    sources.append(row["Article title"])

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))


# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_key = "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"), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)