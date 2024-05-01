# Llama Langchain RAG Project
- **Course**: [CSCI-GA.2565](https://www.sainingxie.com/ml-spring24/)
- **Institution**: New York University
- **Term**: Spring 2024

---

## 🦙💬 Overview

The Llama Langchain RAG project is an innovative application designed specifically for fans of the beloved sitcom [**Friends**](https://en.wikipedia.org/wiki/Friends) for fun. Utilizing the power of Retrieval-Augmented Generation (RAG) coupled with a Language Model (LLM), this project employs [LLaMA 2](https://llama.meta.com/llama2/), finetuned with [Replicate](https://replicate.com/docs/guides/fine-tune-a-language-model) to provide detailed, contextually accurate answers to complex queries related to content, plot, and characters. The app is deployed using [Streamlit](https://streamlit.io/), it includes session chat history and provides an option to select multiple LLaMA2 API endpoints on Replicate. 

**Note:** This is the production version of the application and is optimized for deployment. Running it locally may require modifications to suit the development environment.

---

## Getting Started

### Prerequisites

- Relative API key(s) (optional; e.g. for embedding model)
- Python 3.11 or higher
- Git Large File Storage (LFS) for handling large datasets and model files

### Installation

1. Install dependencies.

   - [Optional but recommended] 
      - Create a virtual python environment with 
         ```
            python -m venv .venv
         ```
      - Activate it with 
         ```
            source .venv/bin/activate
         ```
   - Install dependencies with 
      ```
         pip install -r requirements.txt
      ```

2. Create the Chroma DB:
```
python populate_database.py
```

3. Setup before being able to do inference:

   - Case 1: If you choose to run the selected LLM/Llama 2 locally, you'll need to have [Ollama](https://ollama.com/) installed and run `ollama serve` in a seperate terminal.

   - Case 2: If you choose to do inference with replicate, you'll need to have `REPLICATE_API_TOKEN` setup as an environment variable.

4. Test run to query the Chroma DB:
```
python query_data.py "Which role does Adam Goldberg plays?"
```

5. Start the App:
```
streamlit run app.py
```
 

In case the file size exceeds Github's recommended maximum file size of 50.00 MB, you may need to use [Git Large File Storage](https://git-lfs.github.com).


### Configuration & Features:
1. Finetuning usually involves using a domain related dataset (e.g. [Sujet Finance](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k), [Music-Wiki](https://huggingface.co/datasets/seungheondoh/music-wiki), [MusicPile](https://huggingface.co/datasets/m-a-p/MusicPile?row=29)), in this project, we decided to curate our own (Question-Answer) pairs dataset for training and RAG.
2. Domain-related files (txt or PDFs) are stored in the `data` folder, such as *trivia.txt* and *s1_s2.jsonl*, a vector database is created in `chroma` folder based on the data for RAG. More content could be added as needed.
3. The front-end and deployment is implemented with Streamlit.
4. Chat history is maintained for each session (if you refresh, chat history clears).
5. Option to select between differnet Llama2 chat API endpoints (finetuned-rag, finetuned, rag, base13B).
6. Each model (finetuned-rag, finetuned, rag, base13B) runs on Replicate.

This app was refactored from [dataprofessor's implementation](https://github.com/dataprofessor/llama2/tree/master) and [a16z's implementation](https://github.com/a16z-infra/llama2-chatbot) of their LLaMA2 Chatbot.


### Resources:
- [LLAMA-3 🦙: EASIET WAY To FINE-TUNE ON YOUR DATA 🙌](https://www.youtube.com/watch?v=aQmoog_s8HE)
- [Fine-Tuning LLaMA 2: A Step-by-Step Guide to Customizing the Large Language Model](https://www.datacamp.com/tutorial/fine-tuning-llama-2)
- [RAG + Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I)
- [Building a RAG application from scratch using Python, LangChain, and the OpenAI API](https://www.youtube.com/watch?v=BrsocJb-fAo&t=3685s)
- [Hugging Face + Langchain in 5 mins | Access 200k+ FREE AI models for your AI apps](https://www.youtube.com/watch?v=_j7JEDWuqLE&list=PLz-AnbJcjdrB76ziX7ciillmmBdi0IhvH&index=2)
- [RAG系统：数据越多效果越好吗？](https://github.com/netease-youdao/QAnything/wiki/RAG%E7%B3%BB%E7%BB%9F%EF%BC%9A%E6%95%B0%E6%8D%AE%E8%B6%8A%E5%A4%9A%E6%95%88%E6%9E%9C%E8%B6%8A%E5%A5%BD%E5%90%97%EF%BC%9F)
- [🦙💬 Llama 2 Chat](https://github.com/dataprofessor/llama2?tab=readme-ov-file)
- [Deploy with Streamlit: Secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Streamlit Cheatsheet](https://docs.streamlit.io/develop/quick-reference/cheat-sheet)