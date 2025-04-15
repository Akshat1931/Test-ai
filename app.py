import gradio as gr
import requests
import os
import json

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

def query(payload):
    try:
        # Make sure the payload is formatted correctly for this specific model
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        
        # Debug the raw response if needed
        # print("Raw response:", result)
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if "generated_text" in result[0]:
                return result[0]["generated_text"]
            else:
                return str(result[0])
        else:
            return str(result)
    except requests.exceptions.HTTPError as e:
        error_text = f"HTTP Error: {e}"
        try:
            error_details = response.json()
            error_text += f"\nDetails: {json.dumps(error_details)}"
        except:
            pass
        return error_text
    except Exception as e:
        return f"Error: {str(e)}"

def chat_with_model(message, history):
    # The expected payload format for Mistral-7B-Instruct
    payload = {
        "inputs": message,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    
    response = query(payload)
    
    # If the response includes the original prompt, strip it out
    if response and response.startswith(message):
        response = response[len(message):].strip()
        
    return response

# Create and launch the Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_model,
    title="Mistral-7B Chat",
    description="Chat with Mistral-7B-Instruct using Hugging Face API",
    examples=["Tell me a short story", "What are the benefits of exercise?"],
    cache_examples=False
)

# For Hugging Face Spaces
if __name__ == "__main__":
    demo.launch()