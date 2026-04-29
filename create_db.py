from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. LOAD: Read your info file
print("--- Loading data ---")
loader = TextLoader("C:/Users/santh/python/my_info.txt")
documents = loader.load()

# 2. CHUNK: Use the skill we learned yesterday
print("--- Chunking data ---")
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. EMBED: Turn text into numbers (This part uses the math engine)
# This will download a small model (~100MB) the first time you run it
print("--- Turning text into math (Embeddings) ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. STORE: Save into a local FAISS database
print("--- Saving to local database folder ---")
vector_db = FAISS.from_documents(docs, embeddings)
vector_db.save_local("C:/Users/santh/python/my_vector_db")

print("\nSUCCESS! Your local AI database is created in the 'my_vector_db' folder.")
