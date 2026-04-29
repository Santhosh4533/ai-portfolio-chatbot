import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os
# This looks for the key in the system settings instead of the file
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") 


# 2. LOAD DATABASE
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.load_local(
    "C:/Users/santh/python/my_vector_db", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 3. BRAIN
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

# 4. UI SETUP
st.set_page_config(page_title="Santhosh's AI Assistant", page_icon="🤖")
st.header("Santhosh's Career Assistant (with Memory) 🧠")

# Initialize Chat History in Streamlit Session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. THE CHAT LOOP (Where the memory logic lives)
if user_input := st.chat_input("Ask me about Santhosh..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    # --- MEMORY LOGIC START ---
    # We grab the last 3 exchanges to give the AI "Short-term Memory"
    chat_history_text = ""
    for msg in st.session_state.messages[-4:-1]: # Exclude the current input
        chat_history_text += f"{msg['role'].upper()}: {msg['content']}\n"
    # --- MEMORY LOGIC END ---

    # SEARCH: Find relevant info in your info.txt
    docs = vector_db.similarity_search(user_input, k=2)
    context = "\n".join([d.page_content for d in docs])
    
    # ADVANCED PROMPT: Sending Context + History + New Question
    persona_prompt = f"""
    You are a professional assistant for Santhosh. 
    
    CHAT HISTORY:
    {chat_history_text}
    
    NEW DATA CONTEXT:
    {context}
    
    USER QUESTION: {user_input}
    
    INSTRUCTION: Use the History to understand pronouns like 'he' or 'it'. 
    Use the Context to give factual answers. Be brief.
    """
    
    with st.spinner("Thinking..."):
        response = llm.invoke(persona_prompt)
        answer = response.content
        
        # Display and save AI response
        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
