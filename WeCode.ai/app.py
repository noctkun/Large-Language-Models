# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('WeCODE.ai - A Python Code Generator')
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write a python program on {topic}'
)

# script_template = PromptTemplate(
#     input_variables = ['title'], 
#     template='write me the python program and code for this title TITLE: {title}'
# )

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
# script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, memory=title_memory)
# script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)


# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    # script = script_chain.run(title=title)

    st.write(title) 
    # st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    # with st.expander('Script History'): 
    #     st.info(script_memory.buffer)
