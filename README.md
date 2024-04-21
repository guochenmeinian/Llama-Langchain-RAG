# Llama Langchain RAG Project
- **Course**: [CSCI-GA.2565](https://www.sainingxie.com/ml-spring24/)
- **Institution**: New York University
- **Term**: Spring 2024

---

Install dependencies.

```python
pip install -r requirements.txt
```

1. Create the Chroma DB:
```
python populate_database.py
```

2. Run the selected LLM locally on a seperate terminal:
```
ollama serve
```

3. Query the Chroma DB:
```
python query.py "How to play Monopoly according to the PDF source?"
```

You'll also need to set up an OpenAI account (and set the OpenAI key in your .env as an environment variable) and download Ollama as well as a LLM for this to work.
