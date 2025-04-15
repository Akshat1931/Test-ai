import gradio as gr
import requests
import os
import json
from typing import List, Dict, Any

# Configuration - Using Flan-T5-Base which is smaller but still good for educational purposes
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

def get_token():
    return os.getenv('HF_TOKEN')

def query(message, history: List[Dict[str, str]] = None):
    token = get_token()
    if not token:
        return "Error: API token not configured. Please contact the administrator."
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create a context from history to give the model more context
    context = ""
    if history and len(history) > 0:
        # Format last few exchanges to provide context
        context_entries = history[-3:] if len(history) > 3 else history
        for entry in context_entries:
            if "role" in entry and "content" in entry:
                if entry["role"] == "user":
                    context += f"Question: {entry['content']}\n"
                else:
                    context += f"Answer: {entry['content']}\n"
    
    # Format an educational-focused prompt
    prompt = f"{context}Question: {message}\nAnswer:"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            if "generated_text" in result[0]:
                return result[0]["generated_text"].strip()
            else:
                return str(result[0])
        else:
            return str(result)
    except requests.exceptions.HTTPError as e:
        error_text = f"Error: Unable to generate response. Please try again later."
        print(f"HTTP Error: {e}")
        try:
            error_details = response.json()
            print(f"Details: {json.dumps(error_details)}")
        except:
            pass
        return error_text
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Sorry, there was a problem generating a response. Please try again."

def chat_with_model(message, history):
    # Convert history to our format for context
    formatted_history = []
    for i in range(0, len(history), 2):
        if i < len(history):
            formatted_history.append({"role": "user", "content": history[i]})
        if i+1 < len(history):
            formatted_history.append({"role": "assistant", "content": history[i+1]})
    
    # Get response including context from previous exchanges
    response = query(message, formatted_history)
    return response

# Create a customized interface for educational purposes
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Educational AI Assistant")
    gr.Markdown("Ask questions about any subject to get detailed, educational responses.")
    
    chatbot = gr.ChatInterface(
        fn=chat_with_model,
        examples=[
            "Explain photosynthesis in simple terms",
            "What are the key events of World War II?",
            "How do I solve quadratic equations?",
            "What is the difference between metaphor and simile?",
            "Explain the basics of machine learning"
        ],
        title="Study Assistant",
        theme="soft"
    )
    
    gr.Markdown("### Embed this chatbot in your website")
    gr.Markdown("Use the following HTML code to embed this chatbot in your website:")
    
    iframe_code = f"""
    <iframe
        src="{os.getenv('SPACE_URL', 'YOUR_HUGGING_FACE_SPACE_URL')}"
        width="100%"
        height="600px"
        style="border: 1px solid #ddd; border-radius: 8px;"
    ></iframe>
    """
    gr.Code(value=iframe_code, language="html")

if __name__ == "__main__":
    # Adding share=True to create a public link
    demo.launch(share=True)