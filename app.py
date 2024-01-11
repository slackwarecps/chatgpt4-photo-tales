from IPython.display import Audio
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.chains import LLMChain
import requests
import os
import streamlit as st
from pathlib import Path
from openai import OpenAI

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def img2text(url):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]["generated_text"]
    print('DESCRICAO DA IMAGEM')
    print('========================')
    print(text)
    return text

# gera a historia a parti de uma descrição de imagem.


def generate_story(scenario):
    user_prompt = scenario
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4", n=3)
    system_template = SystemMessagePromptTemplate.from_template(
        " Voce é um contador de historias Voce pode criar uma historia curta baseada em uma simples narrativa, a história não pode ter mais que 50 palavras. Voce sempre traduz as historias para portugues do Brasil")
    user_template = HumanMessagePromptTemplate.from_template("{user_prompt}")
    template = ChatPromptTemplate.from_messages(
        [system_template, user_template])
    chain = LLMChain(llm=llm, prompt=template)
    story = chain.predict(user_prompt=user_prompt)
    print('HISTORIA')
    print('========================')
    print(story)
    return story


def textoparafalaENG(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


def textoParaFalaOpenAI(texto):
    client = OpenAI()
    speech_file_path = Path(__file__).parent / "audio.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=texto
    )
    response.stream_to_file(speech_file_path)


def main():
    st.set_page_config(page_title="asdasdas", page_icon="ball")
    st.header("asdasdasd")
    uploaded_file = st.file_uploader("escolha uma imagem", type="jpg")
    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, 'wb') as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="asdasdasd", use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        textoParaFalaOpenAI(story)
        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        st.audio("audio.mp3")


if __name__ == "__main__":
    main()
