import streamlit as st
import replicate
import os

from query_data import query_finetuned_model, query_llama2_13B_model, query_llama2_70B_model

from dotenv import load_dotenv
load_dotenv()

# Replicate Credentials
replicate_api = os.getenv('REPLICATE_API_TOKEN')
os.environ['REPLICATE_API_TOKEN'] = replicate_api

# App title
st.set_page_config(page_title="💬 Friends Chatbot")

with st.sidebar:
    st.title('💬 Friends Chatbot')
    st.write('This chatbot is based on a finetuned Llama 2 LLM model with RAG. Feel free to ask for questions related to the comsit "Friends".')
    st.success('API key already provided!', icon='✅')

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)



# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLaMA2 response
# Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    # run selected model 
    output = query_finetuned_model(f"{string_dialogue} {prompt_input} Assistant: ")

    # example code
    #output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ", "temperature":0.1, "top_p":0.9, "max_length":512, "repetition_penalty":1})
    
    return output


# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


