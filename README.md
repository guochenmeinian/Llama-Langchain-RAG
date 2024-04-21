# Llama Langchain RAG Project
- **Course**: [CSCI-GA.2565](https://www.sainingxie.com/ml-spring24/)
- **Institution**: New York University
- **Term**: Spring 2024

---

## Overview

The Llama Langchain RAG project is a specialized tool designed to support the learning and exploration of finance domain as well as understanding and analysis of financial markets. Utilizing the power of Retrieval-Augmented Generation (RAG) coupled with a Language Model (LLM), this project employs LLaMA 3, finetuned on the NYU High Performance Computing (HPC) clusters to provide detailed, contextually accurate answers to complex queries within the finance domain.

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
python query_data.py "How to play Monopoly according to the PDF source?"
```

Ensure your `.env` file is configured with the `OPENAI_API_KEY` set as an environment variable and have Ollama downloaded with a local LLM ready to use for this to work. In case the file size exceeds Github's recommended maximum file size of 50.00 MB, you may need to use [Git Large File Storage](https://git-lfs.github.com).

### Configuration

- Store your finance-related PDFs, such as *options_futures_and_derivatives.pdf* and *principles_of_finance.pdf*, within the `data` directory for RAG. More PDFs will be added as the project progresses.
- The LLaMA 3 model will be finetuned using the [Sujet Finance dataset](https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k) from [HuggingFace](https://huggingface.co/).


### TO-DO:
1. Finetune the model using domain related dataset (e.g. https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k)
2. Create vector database with with pdf(s) within `data` dir for RAG


### Some helpful resources:
- [LLAMA-3 🦙: EASIET WAY To FINE-TUNE ON YOUR DATA 🙌](https://www.youtube.com/watch?v=aQmoog_s8HE)
- [Fine-Tuning LLaMA 2: A Step-by-Step Guide to Customizing the Large Language Model](https://www.datacamp.com/tutorial/fine-tuning-llama-2)
- [RAG + Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I)
- [Building a RAG application from scratch using Python, LangChain, and the OpenAI API](https://www.youtube.com/watch?v=BrsocJb-fAo&t=3685s)


