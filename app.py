import gradio as gr
import requests
import os

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# Securely pull token from environment variable
headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    try:
        return result[0]["generated_text"]
    except:
        return result.get("error", "Sorry, something went wrong.")

def chat_with_model(message, history=[]):
    prompt = message
    reply = query({"inputs": prompt})
    history.append((message, reply))
    return history, history

gr.ChatInterface(chat_with_model, title="My Free AI Chatbot 💬").launch()
