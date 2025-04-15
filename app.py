import gradio as gr
import requests
import os
import json
import time
from typing import List, Dict, Any

# Configuration - Using a model that's definitely small enough and reliable
API_URL = "https://api-inference.huggingface.co/models/distilbert/distilgpt2"

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
    
    # Fixed parameters - removed the problematic return_full_text parameter
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True
        }
    }
    
    # Add retry mechanism for when model is loading
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            
            # Handle 503 (model loading) with retries
            if response.status_code == 503:
                error_json = response.json()
                if "estimated_time" in error_json:
                    wait_time = min(error_json.get("estimated_time", 20), 20)
                    print(f"Model is loading. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    continue
            
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    # Clean up the response to only include the answer portion
                    generated_text = result[0]["generated_text"]
                    # If the response contains the original prompt, strip it out
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    return generated_text
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
                # Don't immediately return - allow retries for some errors
                if response.status_code != 503:  # Don't retry for non-503 errors
                    return error_text
            except:
                pass
            
            # If we've reached max retries, give up
            if attempt == max_retries - 1:
                return error_text
                
        except Exception as e:
            print(f"Error: {str(e)}")
            return "Sorry, there was a problem generating a response. Please try again."
    
    return "The model is taking too long to load. Please try again later."

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
    
    # Show the embedding information only if SPACE_URL is set
    space_url = os.getenv('SPACE_URL')
    if space_url:
        gr.Markdown("### Embed this chatbot in your website")
        gr.Markdown("Use the following HTML code to embed this chatbot in your website:")
        
        iframe_code = f"""
        <iframe
            src="{space_url}"
            width="100%"
            height="600px"
            style="border: 1px solid #ddd; border-radius: 8px;"
        ></iframe>
        """
        gr.Code(value=iframe_code, language="html")

if __name__ == "__main__":
    # Adding share=True to create a public link
    demo.launch(share=True)