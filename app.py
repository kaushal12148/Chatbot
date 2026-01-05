import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import gradio as gr



load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
documents = []
folders = glob.glob("knowledge-base/*")
for folder in folders:
    base_dir = os.path.basename(folder)
    
    loader = DirectoryLoader(
        path = folder,
        glob = "**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding":"utf-8"}
    )
    
    docs = loader.load()
    
    for doc in docs:
        
        doc.metadata["doc-type"] = base_dir
        documents.append(doc)
        
print(f"loaded {len(documents)} files")
# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunks = text_splitter.split_documents(documents)
print(f"docs converted to total {len(chunks)} chunks")
# Embedding
embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
db_name = "vector_store"
if os.path.exists(db_name):
    
    print("loading the existing DB...")
    
    vectorstore = Chroma(
        persist_directory=db_name,
        embedding_function = embedding
    )
    
else:
    
    vectorstore = Chroma.from_documents(
        persist_directory=db_name,
        embedding=embedding,
        documents=chunks
    )
    print(f"vector store created with {vectorstore._collection.count()} documents ")
retriever = vectorstore.as_retriever()
SIMILARITY_THRESHOLD = 1.3  # higher = less relevant
RETRIEVAL_K = 10
SYSTEM_PROMPT = """
You are a knowledgeable and friendly chatbot representing the company InsureLLM,
search for answers in the context if you dont find the answer say no 
context:
{Context}

"""
llm = ChatOpenAI(model="gpt-4o-mini", temperature=.5, api_key=api_key)

def fetch_context(question: str):
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"][0]['text'] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []):
  """
  Answer the given question with RAG.
  Show documents only if similarity is good.
  """

  combined = combined_question(question, history)

  # 🔍 similarity search WITH SCORES
  results = vectorstore.similarity_search_with_score(
      combined,
      k=RETRIEVAL_K
  )

  # 🎯 filter relevant docs
  relevant_docs = [
      doc for doc, score in results
      if score <= SIMILARITY_THRESHOLD
  ]

  # build context ONLY from relevant docs
  context = "\n\n".join(doc.page_content for doc in relevant_docs)

  system_prompt = SYSTEM_PROMPT.format(Context=context)

  messages = [SystemMessage(content=system_prompt)]
  messages.extend(convert_to_messages(history))
  messages.append(HumanMessage(content=question))

  response = llm.invoke(messages)

  return response.content, relevant_docs


# GRADIO UI
# =========================
with gr.Blocks(title="LLerem Knowledge Assistant") as demo:
    # Title / Header
    gr.Markdown(
        """
        # 🧠 Insurellm Knowledge Assistant
        Ask questions about Insurellm.
        """
    )
    with gr.Row():
        # ---- Chat Column ----
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=400)
            user_input = gr.Textbox(
                placeholder="Type your question here...",
                show_label=False
            )
            send_btn = gr.Button("Send")
        # ---- Sidebar Column ----
        with gr.Column(scale=2):
            gr.Markdown("### 📄 Retrieved Documents")
            docs_view = gr.Textbox(
                lines = 30,
                interactive=False
            )
    # Chat handler function
    def chat_wrapper(message, history):
    
        history = history or []

        answer, docs = answer_question(message, history)
    
        history.append({
            "role": "user",
            "content": [{"type": "text", "text": message}]
        })
    
        history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": answer}]
        })

        if docs:
          docs_text = "\n\n---\n\n".join(
        f"**Source {i+1}**\n\n{doc.page_content}"
        for i, doc in enumerate(docs)
         )
        else:
        
         docs_text = "ℹ️ No relevant documents were needed for this answer."
         

        # RETURN: updated chat history, sidebar docs, and clear input box
        return history, docs_text, ""
    send_btn.click(
        chat_wrapper,
        inputs=[user_input, chatbot],
        outputs=[chatbot, docs_view, user_input],  # Clear input after sending
    )
    # ENTER key submit
    user_input.submit(
        chat_wrapper,
        inputs=[user_input, chatbot],
        outputs=[chatbot, docs_view, user_input],  # Clear input after sending
    )
demo.launch()
        
        
            
            





