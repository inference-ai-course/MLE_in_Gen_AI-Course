
# Week 4: Retrieval-Augmented Generation (RAG)

Welcome to **Week 4** of the Machine Learning Engineer in the Generative AI Era series!  
This week, you’ll learn how to supercharge LLMs with *real-time retrieval* over your own documents—making your agent fact-grounded, up-to-date, and actually useful for research.

---

## 📝 Lecture Themes

- What is RAG?  
  Retrieval-Augmented Generation combines the reasoning/generation power of LLMs with real-world data sources.
- Why RAG?  
  - LLMs can hallucinate or be out-of-date; RAG lets your model *cite* your actual docs or knowledge base.
  - It’s the engine for nearly all modern AI copilots and research agents.
- The RAG Pipeline:  
  - **Embedding:** Convert your documents to dense vector representations.
  - **Indexing:** Store vectors in a fast vector database for similarity search.
  - **Retrieval:** Find the most relevant snippets given a question.
  - **Generation:** Augment LLM’s answer with retrieved info (citations, quotes).
- Key concepts:  
  - Embeddings (OpenAI, Hugging Face)
  - Distance metrics (cosine, dot product, Euclidean, Manhattan)
  - Tokenization, filtering, and chunking
  - Vector databases (FAISS, Pinecone, Chroma)
  - RAG/LLM integration (LangChain)
  - Evaluation metrics for retrieval quality

---

## 🖥️ Hands-On Project 4: Your Own RAG-Enabled Resume AI

**Goal:**  
Turn your resume + portfolio into a mini research database and build an agent that can answer questions like "What tech did I use on Project X?" using real text from your docs—not just LLM guesses.

### Steps

1. **Prepare Your Data**  
   - Use your resume (PDF, DOCX) and any supporting docs (portfolio, project notes).
2. **Chunk Your Documents**  
   - Use tools like LangChain’s `RecursiveCharacterTextSplitter` to break into ~500-word, 10% overlapping chunks.
3. **Embed & Index**  
   - Use OpenAI or Hugging Face embedding models to turn text chunks into vectors.
   - Store in FAISS (local, fast, open source).
4. **Build Your RAG Chain**  
   - Use LangChain’s `RetrievalQA` to wire up your index to an LLM (OpenAI or your own vLLM server).
   - Try both CPU and GPU-backed (vLLM + Hugging Face) versions.
5. **Ask Questions!**  
   - Examples:  
     - “What Python projects has [Your Name] contributed to?”  
     - “List my main technical skills and years of experience.”
     - “Which projects included NLP or AI?”
6. **Evaluate**  
   - Try good, bad, and adversarial questions.
   - Does the agent quote your docs, or make things up?
   - How would you improve its accuracy or citations?

### Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Hugging Face Embeddings](https://huggingface.co/sentence-transformers)
- [FAISS](https://faiss.ai/)
- [vLLM](https://docs.vllm.ai/en/latest/)

---

## 🗂️ Example Code (see `class_4.py` and notebook for full version)

```python
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# 1. Load your documents
resume = PyPDFLoader("./resume.pdf").load()
extras = TextLoader("./portfolio_notes.txt").load()
docs = resume + extras

# 2. Chunk the docs
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed & Index
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.from_documents(chunks, embedding_model)

# 4. RetrievalQA chain
agent = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k":3})
)

# 5. Ask a question
agent.run("What Python projects has [Your Name] contributed?")
```

---

## 💡 Best Practices

- Tune chunk size and overlap for your data.
- Try different embedding models for speed vs. accuracy.
- Test with both “seen” and “unseen” questions.
- Evaluate using metrics: context recall, nDCG, chunk attribution, etc.
- Use RAGAS or RAGEval for more advanced evaluation.

---

## 📣 Homework Submission

- Complete the RAG pipeline for your resume agent in the notebook.
- Submit:  
  - Your code/notebook  
  - 2–3 demo Q&A examples
  - Short reflection: What worked? What didn’t? How would you make it better?
- Share best/funniest failure cases on Discord.

---

## 🏁 What’s Next?

**Next week:** Supervised Fine-Tuning (SFT)!  
You’ll move from retrieval to training your own custom LLM with personal data.

---

**References:**  
- [RAG Concepts & Code: LangChain, OpenAI, Hugging Face, FAISS docs]  
- [Lecture slide deck: Week 4 RAG (see repo)]  
- [Sample RAG agents: LlamaIndex, OpenAI cookbook]  

---

*Questions or stuck?*  
Ask in Discord, or ping the instructor for debugging help.

---
