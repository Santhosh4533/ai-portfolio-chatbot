# The new way to import in LangChain v0.2+
from langchain_text_splitters import CharacterTextSplitter

# 1. THE DATA
my_text = "Chennai is a major hub for AI. The OMR corridor has many top companies. To get a high tier job, you need to master the basics of RAG and LLMs. Let's practice cutting this text!"

# 2. THE TOOL
splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=30,
    chunk_overlap=5
)

# 3. THE ACTION
chunks = splitter.split_text(my_text)

# 4. THE RESULT
print(f"\n--- AI Chunking Result ---")
for i, piece in enumerate(chunks):
    print(f"Piece {i+1}: {piece}")
