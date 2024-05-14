import streamlit as st
import replicate
import os

from query_data import query_finetuned_rag, query_finetuned, query_rag, query_base

from dotenv import load_dotenv
load_dotenv()


# App title
st.set_page_config(page_title="💬 Friends Chatbot")

selected_option = 'LLaMA2'

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

with st.sidebar:
    st.title('💬 Friends Chatbot')
    st.write('This chatbot offers several variants of the Llama 2 LLM model (finetuned / RAG). Feel free to ask for questions related to the sitcom "Friends".')
    
    # Obtain Credentials
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY'] # this is for the embedding model
    
    selected_option = st.sidebar.selectbox(
        'Choose a LLaMA2 model:', 
        ['LLaMA2', 'Finetuned LLaMA2', 'LLaMA2 with RAG', 'Finetuned LLaMA2 with RAG'],
        key = 'model',
        on_change=clear_chat_history
    )

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)



# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    
    
    # below is for session messages history; we disabled this feature here for test run
    # for dict_message in st.session_state.messages:
    #    if dict_message["role"] == "user":
    #        string_dialogue += "User: " + dict_message["content"] + "\n\n"
    #    else:
    #        string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    
    
    # run selected model
    # 'LLaMA2', 'Finetuned LLaMA2', 'LLaMA2 with RAG', 'Finetuned LLaMA2 with RAG'
    output = ""
    if selected_option == 'Finetuned LLaMA2 with RAG':
        output = query_finetuned_rag(f"{string_dialogue} {prompt_input} Assistant: ")
    elif selected_option == 'LLaMA2 with RAG':
        output = query_rag(f"{string_dialogue} {prompt_input} Assistant: ")
    elif selected_option == 'Finetuned LLaMA2':
        output = query_finetuned(f"{string_dialogue} {prompt_input} Assistant: ")
    else:
        output = query_base(f"{string_dialogue} {prompt_input} Assistant: ")

    
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


