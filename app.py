import streamlit as st
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(
    page_title="Chat With Your Dataset",
    page_icon="📊",
    layout="centered"
)

# --- Header ---
st.title("📊 Chat With Your Dataset")
st.markdown("Upload a CSV and ask questions about its summary stats!")

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("Settings")
    
    # Provider Selection
    provider = st.radio("Select LLM Provider", ["Google Gemini", "OpenAI / Compatible", "Hugging Face (Free Tier)"])
    
    api_key = ""
    base_url = None
    hf_repo_id = "HuggingFaceH4/zephyr-7b-beta" # Default for HF (Safe, non-gated)
    
    if provider == "Google Gemini":
        api_key = st.text_input("Enter Google API Key", type="password")
        st.markdown("[Get a Google API Key](https://aistudio.google.com/app/apikey)")
        
    elif provider == "OpenAI / Compatible":
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        base_url = st.text_input("Base URL (Optional)", placeholder="https://api.openai.com/v1")
        st.markdown("Leave Base URL empty for standard OpenAI.")
        
    elif provider == "Hugging Face (Free Tier)":
        api_key = st.text_input("Enter Hugging Face Token", type="password")
        st.markdown("[Get a Free HF Token](https://huggingface.co/settings/tokens)")
        hf_repo_id = st.text_input("Model Repo ID", value="HuggingFaceH4/zephyr-7b-beta")
        st.caption("Embeddings run locally (no key needed). LLM runs via HF Inference API.")

# --- Helper Function: Process Data ---
def process_csv(file):
    """
    Loads CSV, generates summary statistics, and returns the DataFrame and summary text.
    """
    try:
        df = pd.read_csv(file)
        
        # 1. Basic Info
        num_rows, num_cols = df.shape
        columns = ", ".join(df.columns.tolist())
        
        # 2. Missing Values
        missing_values = df.isnull().sum()
        missing_str = missing_values[missing_values > 0].to_string()
        if missing_str == "Series([], )":
            missing_str = "None"
            
        # 3. Descriptive Statistics
        stats = df.describe().to_string()
        
        # 4. Construct Summary Text
        summary_text = f"""
Dataset Overview:
- Number of Rows: {num_rows}
- Number of Columns: {num_cols}
- Column Names: {columns}

Missing Values per Column:
{missing_str}

Statistical Summary (Numeric Columns):
{stats}
"""
        return df, summary_text
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None, None

# --- Helper Function: Get LLM & Embeddings ---
def get_llm_resources(provider, api_key, base_url=None, hf_repo_id=None):
    try:
        if provider == "Google Gemini":
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.2)
        
        elif provider == "OpenAI / Compatible":
            # Handle empty base_url
            if not base_url:
                base_url = None # Default to OpenAI
            
            embeddings = OpenAIEmbeddings(api_key=api_key, base_url=base_url)
            llm = ChatOpenAI(api_key=api_key, base_url=base_url, temperature=0.2)
            
        elif provider == "Hugging Face (Free Tier)":
            # Embeddings: Local (cpu) using sentence-transformers
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Base Endpoint
            endpoint = HuggingFaceEndpoint(
                repo_id=hf_repo_id,
                huggingfacehub_api_token=api_key,
                temperature=0.2,
                task="text-generation"
            )
            
            # Conditional Logic: Use Chat wrappers for Instruct/Chat/Zephyr models
            # This ensures roles (System/User) are formatted correctly for the specific model.
            model_name_lower = hf_repo_id.lower()
            if "instruct" in model_name_lower or "chat" in model_name_lower or "zephyr" in model_name_lower:
                from langchain_huggingface import ChatHuggingFace
                llm = ChatHuggingFace(llm=endpoint)
            else:
                llm = endpoint
            
        return embeddings, llm
    except Exception as e:
        st.error(f"Error initializing AI resources: {e}")
        return None, None

# --- Helper Function: Setup QA Chain ---
def setup_qa_chain(summary_text, api_key, provider, base_url=None, hf_repo_id=None):
    """
    Creates a RetrievalQA chain using FAISS and the selected Provider.
    """
    try:
        # 1. Init Resources
        embeddings, llm = get_llm_resources(provider, api_key, base_url, hf_repo_id)
        if not embeddings or not llm:
            return None

        # 2. Create Document
        doc = Document(page_content=summary_text, metadata={"source": "dataset_summary"})
        
        # 3. Vector Store
        vectorstore = FAISS.from_documents([doc], embeddings)
        retriever = vectorstore.as_retriever()
        
        # 4. Prompt Template
        template = """
You are a helpful and precise Data Analyst.
Your goal is to answer questions about a dataset based ONLY on the provided summary context.

Context (Dataset Summary):
{context}

Question:
{question}

Instructions:
1. Answer clearly in student-friendly language.
2. Use the statistical data provided in the context.
3. If the answer is not in the context, state: "I cannot answer this based on the available dataset summary."
4. Do not guess or hallucinate specific row data that is not in the summary.

Answer:
"""
        prompt = ChatPromptTemplate.from_template(template)
        
        # 5. Chain Construction (LCEL)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain
        
    except Exception as e:
        st.error(f"Error setting up Chain: {e}")
        return None

# --- Main App Logic ---

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Process the CSV
    df, summary_text = process_csv(uploaded_file)
    
    if df is not None:
        # Show Preview
        st.subheader("Example Data (First 5 Rows)")
        st.dataframe(df.head())
        
        # Show Raw Summary (Optional, inside expander)
        with st.expander("View Dataset Summary"):
            st.text(summary_text)
            
        # --- Chat Section ---
        st.subheader("Chat with your Data")
        
        if not api_key:
            st.warning(f"⚠️ Please enter your {provider} credentials in the sidebar.")
        else:
            # Generate a unique key for the session state based on configs
            config_key = f"{uploaded_file.name}_{provider}_{api_key[:5]}"
            
            # Initialize Chain
            if 'qa_chain' not in st.session_state or st.session_state.get('config_key') != config_key:
                 with st.spinner(f"Initializing {provider} agent..."):
                    chain = setup_qa_chain(summary_text, api_key, provider, base_url, hf_repo_id)
                    if chain:
                        st.session_state['qa_chain'] = chain
                        st.session_state['config_key'] = config_key
                        st.success("Ready! Ask a question below.")
            
            # Question Input
            user_question = st.text_input("Ask a question about the dataset:")
            
            if user_question:
                if 'qa_chain' in st.session_state:
                    with st.spinner("Thinking..."):
                        try:
                            # Run the chain
                            res = st.session_state['qa_chain'].invoke(user_question)
                            st.write("### Answer:")
                            st.write(res)
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
