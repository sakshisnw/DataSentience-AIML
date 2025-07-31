from deep_translator import GoogleTranslator
import openai
import streamlit as st

def translate_to_english(text , src_lang):
    if src_lang.lower()=="english":
        return text
    return GoogleTranslator(source=src_lang.lower(), target="en").translate(text)

def translate_from_english(text, dest_lang):
    if dest_lang.lower()=="english":
        return text
    return GoogleTranslator(source="en", target=dest_lang.lower()).translate(text)

def get_response_from_openai(prompt):
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "system", "content": "You are a helpful healthcare assistant. Based on the symptoms, suggest possible illness in a practical yet not scary way. Also, mention the urgency level (mild , moderate, severe). If the symptoms are not related to any illness, suggest that the user consult a doctor for further evaluation."},
        {"role": "user", "content": prompt}],
        temperature = 0.7,
    )
    return response.choices[0].message['content']