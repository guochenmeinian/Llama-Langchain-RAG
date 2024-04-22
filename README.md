# Llama Langchain RAG Project
- **Course**: [CSCI-GA.2565](https://www.sainingxie.com/ml-spring24/)
- **Institution**: New York University
- **Term**: Spring 2024

---

## Overview

The Llama Langchain RAG project is a specialized tool designed to support the learning and exploration of music relam. Utilizing the power of Retrieval-Augmented Generation (RAG) coupled with a Language Model (LLM), this project employs LLaMA 3, finetuned on the NYU High Performance Computing (HPC) clusters to provide detailed, contextually accurate answers to complex queries related to music theory, history, and analysis.

---

## Getting Started

### Prerequisites

- Access to NYU's HPC clusters (optional)
- An OpenAI account with an API key (optional: for embedding model)
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

3. Run the selected LLM locally on a seperate terminal:
```
ollama serve
```

4. Query the Chroma DB:
```
python query_data.py "What is chord melody?"
```

Ensure your `.env` file is configured with the `OPENAI_API_KEY` set as an environment variable and have Ollama downloaded with a local LLM ready to use for this to work. In case the file size exceeds Github's recommended maximum file size of 50.00 MB, you may need to use [Git Large File Storage](https://git-lfs.github.com).


### Configuration (TO-DO):
1. Finetune the LLaMA 3 model using domain related dataset (e.g. [Sujet Finance](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k), [Music-Wiki](https://huggingface.co/datasets/seungheondoh/music-wiki), [MusicPile](https://huggingface.co/datasets/m-a-p/MusicPile?row=29))
2. Store your domain-related files (currently only support PDFs), such as *Basic Music Theory.pdf* and *Blues The Basics.pdf*, vector database will be created with PDFs within `data` dir for RAG. More PDFs will be added as the project progresses.


### Some helpful resources:
- [LLAMA-3 🦙: EASIET WAY To FINE-TUNE ON YOUR DATA 🙌](https://www.youtube.com/watch?v=aQmoog_s8HE)
- [Fine-Tuning LLaMA 2: A Step-by-Step Guide to Customizing the Large Language Model](https://www.datacamp.com/tutorial/fine-tuning-llama-2)
- [RAG + Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I)
- [Building a RAG application from scratch using Python, LangChain, and the OpenAI API](https://www.youtube.com/watch?v=BrsocJb-fAo&t=3685s)
- [Hugging Face + Langchain in 5 mins | Access 200k+ FREE AI models for your AI apps](https://www.youtube.com/watch?v=_j7JEDWuqLE&list=PLz-AnbJcjdrB76ziX7ciillmmBdi0IhvH&index=2)


