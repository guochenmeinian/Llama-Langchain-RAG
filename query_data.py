import os
import argparse
import replicate

from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_llama2_13B_model(query_text) 


# specially fine-tuned for our task
# Note: need `REPLICATE_API_TOKEN` as an environment variable
def query_finetuned_model(query_text: str):
    
    training = replicate.trainings.get("3ac8b8mygxrgg0cf4dcvh6qwmg")

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response_text = replicate.run(
      training.output["version"],
      input={"prompt": prompt, "stop_sequences": "</s>"}
    )
    
    output = "".join(response_text)  # Assuming response_text is iterable and contains strings.
    print(f"{output}\nSources: {sources}")

    return output


# base llama2 model with 70B parameters
# Note: need `REPLICATE_API_TOKEN` as an environment variable
def query_llama2_70B_model(query_text: str):
    
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    output = ""

    # The meta/llama-2-70b-chat model can stream output as it's running.
    for event in replicate.stream(
        "meta/llama-2-70b-chat",
        input={
            "top_k": 0,
            "top_p": 1,
            "prompt": prompt,
            "temperature": 0.5,
            "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
            "length_penalty": 1,
            "max_new_tokens": 500,
            "min_new_tokens": -1,
            "prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
            "presence_penalty": 0
        },
    ):
        output += str(event)
    
    print(f"{output}\nSources: {sources}")

    return output


# base llama2 model with 13B parameters
# Note: need `REPLICATE_API_TOKEN` as an environment variable
def query_llama2_13B_model(query_text: str):
    
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # The meta/llama-2-13b-chat model can stream output as it's running.
    for event in replicate.stream(
        "meta/llama-2-13b-chat",
        input={
            "top_k": 0,
            "top_p": 1,
            "prompt": prompt,
            "temperature": 0.75,
            "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
            "length_penalty": 1,
            "max_new_tokens": 500,
            "prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
            "presence_penalty": 0
        },
    ):
        print(str(event), end="")
      
    print(f"\nSources: {sources}")


def query_local_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    #print(prompt)

    # https://ollama.com/library/llama3:8b
    model = Ollama(model="llama3:8b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()

