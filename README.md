# Llama Langchain RAG Project
- **Course**: [CSCI-GA.2565](https://www.sainingxie.com/ml-spring24/)
- **Institution**: New York University
- **Term**: Spring 2024

---

## Overview

The Llama Langchain RAG project is a specialized tool designed to answer questions related to the sitcom [**Friends**](https://en.wikipedia.org/wiki/Friends) for fun. Utilizing the power of Retrieval-Augmented Generation (RAG) coupled with a Language Model (LLM), this project employs [LLaMA 3](https://llama.meta.com/llama3/), finetuned with [replicate](https://replicate.com/docs/guides/fine-tune-a-language-model) to provide detailed, contextually accurate answers to complex queries related to content, plot, and characters.

---

## Getting Started

### Prerequisites

- Relative API key(s) (optional; e.g. for embedding model)
- Python 3.9 or higher
- Git Large File Storage (LFS) for handling large datasets and model files

### Installation

1. Install dependencies.
```
pip install -r requirements.txt
```

2. Create the Chroma DB:
```
python populate_database.py
```

3. Run the selected LLM locally on a seperate terminal (Optional: if used Llama 3 locally): 
```
ollama serve
```

4. Query the Chroma DB:
```
python query_data.py "Which role does Adam Goldberg plays?"
```

In case the file size exceeds Github's recommended maximum file size of 50.00 MB, you may need to use [Git Large File Storage](https://git-lfs.github.com).


### Configuration (TO-DO):
1. Finetune the LLaMA 3 model using domain related dataset (e.g. [Sujet Finance](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k), [Music-Wiki](https://huggingface.co/datasets/seungheondoh/music-wiki), [MusicPile](https://huggingface.co/datasets/m-a-p/MusicPile?row=29))
2. Store your domain-related files (txt or PDFs) in the `data` folder, such as *QA.txt* and *The Basics.pdf*, a vector database will be created within `chroma` folder for RAG. More content will be added as the project progresses.


### Some helpful resources:
- [LLAMA-3 🦙: EASIET WAY To FINE-TUNE ON YOUR DATA 🙌](https://www.youtube.com/watch?v=aQmoog_s8HE)
- [Fine-Tuning LLaMA 2: A Step-by-Step Guide to Customizing the Large Language Model](https://www.datacamp.com/tutorial/fine-tuning-llama-2)
- [RAG + Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I)
- [Building a RAG application from scratch using Python, LangChain, and the OpenAI API](https://www.youtube.com/watch?v=BrsocJb-fAo&t=3685s)
- [Hugging Face + Langchain in 5 mins | Access 200k+ FREE AI models for your AI apps](https://www.youtube.com/watch?v=_j7JEDWuqLE&list=PLz-AnbJcjdrB76ziX7ciillmmBdi0IhvH&index=2)
- [RAG系统：数据越多效果越好吗？](https://github.com/netease-youdao/QAnything/wiki/RAG%E7%B3%BB%E7%BB%9F%EF%BC%9A%E6%95%B0%E6%8D%AE%E8%B6%8A%E5%A4%9A%E6%95%88%E6%9E%9C%E8%B6%8A%E5%A5%BD%E5%90%97%EF%BC%9F)


