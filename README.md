# Llama Langchain RAG Project
- **Course**: [CSCI-GA.2565](https://www.sainingxie.com/ml-spring24/)
- **Institution**: New York University
- **Term**: Spring 2024

---

The Llama Langchain RAG project is a specialized tool designed to support the learning and exploration of a specific domain (TBD). Utilizing the power of Retrieval-Augmented Generation (RAG) with a Language Model (LLM), the finetuned model provides detailed, contextually accurate answers to queries.

---

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
python query.py "How to play Monopoly according to the PDF source?"
```

You'll also need to set up an OpenAI account (and set the OpenAI key in your .env as an environment variable) and download Ollama as well as a LLM for this to work.

### Next Step:
1. Finetune the model using domain related dataset (e.g. https://huggingface.co/datasets/sujet-ai/Sujet-Finance-Instruct-177k)
2. Create vector database with with files/pdf within `data` dir for RAG


### Some Helpful Tutorials for training:
- [LLAMA-3 🦙: EASIET WAY To FINE-TUNE ON YOUR DATA 🙌](https://www.youtube.com/watch?v=aQmoog_s8HE)
- [Fine-Tuning LLaMA 2: A Step-by-Step Guide to Customizing the Large Language Model](https://www.datacamp.com/tutorial/fine-tuning-llama-2)
- [RAG + Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I)


