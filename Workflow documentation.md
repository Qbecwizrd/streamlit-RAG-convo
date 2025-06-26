# Documentation: Conversational RAG Chatbot with PDF Uploads

## 🧠 Project Overview

This project implements a **Conversational Retrieval-Augmented Generation (RAG)** system using:

* 🗂️ PDF uploads
* 🤖 Chat-based interaction
* 🧠 Memory-aware responses
* 🔎 Vector similarity search
* 🧾 Stateful chat sessions

It uses **LangChain**, **Hugging Face embeddings**, **Groq LLMs**, and **Streamlit** for UI.

---

## 🔗 Major Workflows

### 1. 📄 PDF Loading & Chunking

**Goal:** Ingest user-uploaded PDFs and prepare them for semantic search.

**Steps:**

* Upload PDFs using `st.file_uploader()`.
* Save temporarily to disk.
* Load content using `PyPDFLoader`.
* Split long documents into chunks using `RecursiveCharacterTextSplitter`.

```python
loader = PyPDFLoader(temp_pdf)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
splits = text_splitter.split_documents(docs)
```

---

### 2. 🧠 Embedding & Vector Store

**Goal:** Convert text into vectors for semantic similarity search.

**Steps:**

* Use `HuggingFaceEmbeddings` with a valid model like `all-MiniLM-L6-v2`.
* Store embeddings using `Chroma`.
* Convert to retriever with `.as_retriever()`.

```python
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstoredb = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstoredb.as_retriever()
```

---

### 3. 🧩 History-Aware Retriever

**Goal:** Reformulate user questions using chat context.

**Steps:**

* Create a custom system prompt to guide reformulation.
* Use `ChatPromptTemplate` + `MessagesPlaceholder` for context.
* Apply `create_history_aware_retriever()` to combine retriever and prompt.

```python
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ('system', contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
```

---

### 4. 📚 QA Chain (StuffDocuments)

**Goal:** Generate answers using retrieved documents.

**Steps:**

* Use another `ChatPromptTemplate` to define how the model should answer.
* Combine this with `create_stuff_documents_chain()`.

```python
qa_prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
```

---

### 5. 🔄 Retrieval Chain

**Goal:** Connect history-aware retriever and QA chain.

**Steps:**

* Use `create_retrieval_chain()` to stitch together retriever + QA.

```python
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

---

### 6. 💬 Chat History + RAG Chain

**Goal:** Maintain chat session memory across turns.

**Steps:**

* Store sessions using `st.session_state.store`.
* Use `ChatMessageHistory` per session.
* Wrap RAG chain with `RunnableWithMessageHistory`.

```python
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)
```

---

### 7. 🖼️ Streamlit Frontend

**Goal:** Provide an interactive UI.

**UI Components:**

* `st.title()` and `st.write()` for headers
* `st.text_input()` for API key, session ID, and user input
* `st.file_uploader()` for uploading PDFs
* Display chat and assistant output

---

## ✅ Example Supported Models (Groq)

Use one of the following for `ChatGroq`:

* `'mixtral-8x7b-32768'`
* `'llama2-70b-4096'`
* `'llama-3.1-8b-instant'`
* `'gemma2-9b-it'` *(check access)*

```python
llm = ChatGroq(groq_api_key=api_key, model_name='mixtral-8x7b-32768')
```

---

## ✅ Hugging Face Embedding Models

Valid for `HuggingFaceEmbeddings`:

* `'sentence-transformers/all-MiniLM-L6-v2'`
* `'sentence-transformers/all-mpnet-base-v2'`
* `'paraphrase-MiniLM-L6-v2'`

---

## 🧪 Testing Tips

* Ensure Hugging Face token is set in `.env` or passed explicitly
* Validate Groq model names and usage limits via their console
* Use `st.write()` or logs to debug inputs/outputs

---

## 📦 Dependencies

* `langchain`
* `langchain-core`
* `langchain-community`
* `langchain-chroma`
* `langchain-groq`
* `sentence-transformers`
* `streamlit`
* `python-dotenv`

---

## 📁 File Structure

```
📦project_folder
 ┣ 📄 app.py              ← Streamlit app
 ┣ 📄 .env                ← Contains HF_TOKEN and other secrets
 ┗ 📁 test/               ← venv folder (optional)
```

---

## 📌 Conclusion

This project showcases an end-to-end Conversational RAG pipeline using PDFs, embeddings, LLMs, and memory-aware interactions.
Perfect for building knowledge assistants, customer support tools, or personalized tutors.

Let me know if you'd like diagrams or Docker setup too!
