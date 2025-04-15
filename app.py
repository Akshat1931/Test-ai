import gradio as gr
import requests
import os

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        result = response.json()
        
        # Handle the response structure correctly
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            return str(result)  # Return the raw response as string for debugging
    except Exception as e:
        return f"Error: {str(e)}"

def chat_with_model(message, history):
    # Format the conversation history properly for the model
    # Mistral-7B-Instruct typically expects a specific format for instructions
    formatted_prompt = f"<s>[INST] {message} [/INST]"
    
    # Get response from the model
    response = query({"inputs": formatted_prompt})
    
    # Clean up the response if needed
    if response.startswith(formatted_prompt):
        response = response[len(formatted_prompt):].strip()
    
    return response

# Create and launch the Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_model,
    title="Mistral-7B Chat",
    description="Chat with Mistral-7B-Instruct model from Hugging Face",
    examples=["Hello, how are you?", "Explain quantum computing in simple terms"],
    cache_examples=False
)

# For Hugging Face Spaces, we don't need the share=True parameter
if __name__ == "__main__":
    demo.launch()