"""
This is a script that will generate an interview style podcast given a topic.
It will use GPT-3 via langchain agents to generate a script, and then use ElevenLabs to generate an audio file from the script.
to start it will use pyttsx3 to generate the audio and once I have it running I will switch to ElevenLabs

It borrows heavily from langchain documentation and the RAVEN project by david shapiro (https://github.com/daveshap/raven)
"""

# imports
import os
from dataclasses import dataclass

import pyttsx3
import json
import datetime
import requests
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain import text_splitter

# set global variables
VERBOSE = True
HOST_TEMPLATE = """I am a chatbot named HOST. I am a podcast host who is interested in learning as much as I can about 
my current GUEST. I am an accomplished journalist who is known for asking guests tough questions and challenging the 
beliefs and assumptions	of the GUEST that I am interviewing. I am skeptical about anything I read and only believe 
things that are known facts. I don't believe conspiracy theories. I use the socratic method to expose flaws in logic.

I am a very good listener and will ask follow up questions to clarify the GUEST's answers. I will ask questions that
challenge the GUEST's beliefs and assumptions. I will ask questions that are designed to get the GUEST to reveal
their biases and prejudices. I will ask questions that are designed to get the GUEST to reveal their opinions on
controversial topics. 
"""

GUEST_TEMPLATE = """I am a chatbot named GUEST. My goals are to share knowledge about myself with the world by 
answering the questions I am asked by the HOST. I am very opinionated and adversarial, and some of my opinions are 
controversial. I am funny and a little bit sarcastic in my answers, and I have a tendency to ramble and get off 
topic. I will avoid repeating any phrase that has already been used, and find interesting ways to rephrase 
statements. I believe everything that I say, and will not back down from by beliefs no mater how much I am challenged.
"""

TOPIC_LIST = ['climate change', 'the future of the world', 'the coming AI singularity', 'post-scarcity society']
# set API keys
with open('/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt', 'r') as f:
    os.environ["OPENAI_API_KEY"] = f.read()


# utility functions
def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def speak(text, voice):
    voice.say(text)
    voice.runAndWait()


# initialize stuff
def init():
    vprint('Initializing...')
    llm = init_llm()
    vprint('LLM initialized')
    tools = init_tools()
    vprint('Tools initialized')
    agent = init_agent(llm, tools)
    vprint('Agent initialized')
    voice = init_voice()
    vprint('Voice initialized')
    return llm, tools, agent, voice


def init_llm():
    llm = OpenAI(temperature=0.7)
    return llm


def init_tools():
    tools = load_tools([])
    return tools


def init_memory():
    memory = ConversationBufferMemory(memory_key="chat_history")
    return memory


def init_agent(llm, tools):
    memory = init_memory()
    agent = initialize_agent(llm=llm, tools=tools, agent='conversational-react-description', memory=memory)
    return agent


def init_voice():
    voice = pyttsx3.init()
    return voice


def generate_response(agent, text):
    try:
        response = {agent.run(input=text)}
    except Exception as e:
        response = f'<Error>: {e}'
    return response


def generate_questions(llm, topic_list):
    vprint('Generating questions...')
    questions = []
    template = """
    {host_template}
    I am interviewing a guest about {topic} and I am going to write a list of questions to ask the guest. 
    
    
    The format of the questions should be. the questions should not be numbered, but should be in a bulleted list format as follows:
    - What do you feel about the current state of the world?
    - What future do you see for the world?
    - Is it concerning to you that the world is changing so fast?
    
    QUESTIONS:
    """

    for topic in topic_list:
        prompt = PromptTemplate(input_variables=['host_template', 'topic'],
                                template=template
                                )
        question = llm(prompt.format(host_template=HOST_TEMPLATE,
                                     topic=topic)
                       )

        question_list = question.split('\n')
        for q in question_list:
            vprint(f'generated question: {q}')
            questions.append(q)
    vprint('Questions generated')
    for question in questions:
        vprint(question)
    return questions


def ask_follow_up_questions():
    # TODO: make a Tool for the host agent that will ask follow up questions if necessary
    pass


def interview(questions, llm):
    q_and_a = {}
    h_template = """
    {host_template}
    I am interviewing a guest and I am going to phrase the following question in a conversational way:
    {question}
    """
    g_template = """
    {guest_template}
    I am being interviewed by a host and I am going to answer the following question:
    
    {question}
    """

    for question in questions:
        vprint(f'generating question and answer for {question}')
        host_prompt = PromptTemplate(input_variables=['host_template', 'question'],
                                     template=h_template, )
        host_prompt = host_prompt.format(host_template=HOST_TEMPLATE,
                                         question=question, )
        host_question = llm(host_prompt)
        vprint(f'generated host question: {host_question}')
        guest_prompt = PromptTemplate(input_variables=['guest_template', 'question'],
                                      template=g_template)
        guest_prompt = guest_prompt.format(guest_template=GUEST_TEMPLATE,
                                           question=host_question)
        guest_response = llm(guest_prompt)
        q_and_a[host_question] = guest_response
        vprint(f'generated guest response: {guest_response}')
    return q_and_a


def main():
    llm, tools, agent, engine = init()
    questions = generate_questions(llm, TOPIC_LIST)
    q_and_a = interview(questions, llm)
    voice_list = engine.getProperty('voices')
    for q, a in q_and_a.items():
        engine.setProperty('voice', voice_list[0].id)
        speak(q, engine)
        engine.setProperty('voice', voice_list[10].id)
        speak(a, engine)


if __name__ == '__main__':
    main()
