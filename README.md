LLerem Knowledge Assistant

LLerem Knowledge Assistant is a Retrieval-Augmented Generation (RAG) chatbot built to answer questions using a private knowledge base. It combines semantic search with conversational memory to deliver accurate, context-aware responses while keeping hallucinations in check.

The chatbot indexes Markdown documents into a vector database and retrieves only the most relevant content for each query. It supports conversation-aware retrieval, meaning previous user questions influence document search, while each response is grounded in freshly selected context.

✨ Key Features

🔍 Semantic search with similarity scoring using Chroma

🧩 Conversation-aware retrieval across multiple turns

📄 Dynamic context construction from relevant documents only

🤖 LLM-powered responses via OpenAI models

🧠 Clear separation of chat memory and document evidence

🖥️ Interactive Gradio UI with retrieved document preview

📦 Local vector store persistence

🛠️ Tech Stack

LangChain

ChromaDB

HuggingFace Embeddings (MiniLM)

OpenAI GPT models

Gradio

Python

🎯 Use Cases

Internal company knowledge assistants

Documentation Q&A bots

Private RAG-based chat systems

Learning and experimentation with modern RAG architectures

This project demonstrates a production-style RAG pipeline, emphasizing relevance filtering, controlled context injection, and conversational coherence.
