from deep_translator import GoogleTranslator
import google.generativeai as genai
import streamlit as st

def translate_to_english(text , src_lang):
    if src_lang.lower()=="english":
        return text
    return GoogleTranslator(source=src_lang.lower(), target="en").translate(text)

def translate_from_english(text, dest_lang):
    if dest_lang.lower()=="english":
        return text
    return GoogleTranslator(source="en", target=dest_lang.lower()).translate(text)

def get_response_from_gemini(user_message, model):
    if model is None:
        return "I'm sorry, I can't connect right now. Please check the API configuration."

    prompt = f"""
    You are a helpful healthcare assistant. Based on the symptoms, suggest possible illness in a practical yet non scary way. Also, mention the urgency level (mild , moderate, severe). If the symptoms are not related to any illness, suggest that the user consult a doctor for further evaluation.
    
    User message: {user_message}
    
    Respond in a honest, no fluff manner. Also, do check all posibilites before concluding.
    """
    try:
        response = model.generate_content(prompt)
        # Clean the response to remove any HTML or unwanted formatting
        cleaned_response = clean_ai_response(response.text)
        return cleaned_response
    except Exception as e:
        return "Trouble connecting. Please try again later."

