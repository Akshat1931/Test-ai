import gradio as gr
import requests
import os
import json
import time
import hashlib
from functools import lru_cache
from typing import List, Dict, Any
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
# Using Microsoft's Phi-3-mini (2.8B), one of the most capable small models available
API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"

# Cache configuration
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Set up a semaphore to limit concurrent API calls
API_SEMAPHORE = threading.Semaphore(3)

def get_token():
    token = os.getenv('HF_TOKEN')  # Make sure this matches your Render secret name
    if not token:
        logger.warning("HF_TOKEN environment variable not set")
    return token

def get_cached_response(prompt_hash):
    """Get a cached response if available"""
    cache_file = os.path.join(CACHE_DIR, f"{prompt_hash}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                # Check if cache entry is expired (older than 7 days)
                if time.time() - data.get("timestamp", 0) < 7 * 24 * 60 * 60:
                    return data.get("response")
                else:
                    logger.info(f"Cache entry expired for {prompt_hash}")
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
    return None

def save_to_cache(prompt_hash, response):
    """Save a response to cache with timestamp"""
    cache_file = os.path.join(CACHE_DIR, f"{prompt_hash}.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                "response": response,
                "timestamp": time.time()
            }, f)
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")

def format_phi3_prompt(message, history):
    """Format the prompt for Phi-3"""
    # Start with a system prompt to guide model behavior
    prompt = "<|system|>\nYou are a helpful, accurate, and educational AI assistant. You provide informative, concise, and helpful responses to questions on any academic or educational topic. You aim to explain concepts clearly and help students learn effectively.\n\n"
    
    # Add conversation history
    for i in range(0, len(history), 2):
        if i < len(history):
            prompt += f"<|user|>\n{history[i]}\n\n"
        if i+1 < len(history):
            prompt += f"<|assistant|>\n{history[i+1]}\n\n"
    
    # Add the current message
    prompt += f"<|user|>\n{message}\n\n<|assistant|>\n"
    
    return prompt

def query(message, history=None):
    with API_SEMAPHORE:
        token = get_token()
        if not token:
            return "Error: API token not configured. Please set the HF_TOKEN environment variable."
        
        # Format the prompt according to Phi-3's expected format
        prompt = format_phi3_prompt(message, history if history else [])
        
        # Generate a unique hash for this prompt to use as cache key
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        # Check cache first
        cached_response = get_cached_response(prompt_hash)
        if cached_response:
            logger.info("Using cached response")
            return cached_response
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Parameters optimized for Phi-3-mini
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.15,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        max_retries = 3
        backoff_factor = 2
        
        for attempt in range(max_retries):
            try:
                # Add a timeout to prevent hanging
                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                
                # Special handling for model loading status
                if response.status_code == 503:
                    error_json = response.json()
                    wait_time = min(error_json.get("estimated_time", 20), 30) * backoff_factor ** attempt
                    logger.info(f"Model is loading. Waiting {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                # Extract the generated text
                if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                    generated_text = result[0]["generated_text"].strip()
                    
                    # Clean up any unwanted tokens
                    if "<|assistant|>" in generated_text:
                        generated_text = generated_text.split("<|assistant|>")[1].strip()
                    
                    # Remove any trailing system tokens
                    for token in ["<|user|>", "<|system|>", "<|end|>"]:
                        if token in generated_text:
                            generated_text = generated_text.split(token)[0].strip()
                    
                    # Cache the successful response
                    save_to_cache(prompt_hash, generated_text)
                    return generated_text
                else:
                    # Return whatever we got
                    text_response = str(result)
                    logger.warning(f"Unexpected response format: {text_response[:100]}...")
                    return text_response
                    
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP Error: {e}")
                try:
                    error_details = response.json()
                    logger.error(f"API Error Details: {error_details}")
                    
                    # Don't retry for certain error types
                    if response.status_code != 503:
                        return f"Error: Unable to generate response. Please try again later. (Error: {response.status_code})"
                except:
                    pass
                
                if attempt == max_retries - 1:
                    return "Error: Maximum retry attempts reached. Please try again later."
                    
            except requests.exceptions.Timeout:
                logger.error("Request timed out")
                if attempt == max_retries - 1:
                    return "The request timed out. The service might be experiencing high load."
                time.sleep(5 * (attempt + 1))  # Progressive backoff
                
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return f"Sorry, there was a problem generating a response: {str(e)}"
        
        return "The model is taking too long to load. Please try again later."

def chat_with_model(message, history):
    """Handle chat interactions with proper history formatting"""
    # Convert the history to the expected format
    flat_history = []
    if isinstance(history, list):
        if len(history) > 0:
            if isinstance(history[0], dict):
                # New format with dictionaries
                for msg in history:
                    flat_history.append(msg["content"])
            elif isinstance(history[0], tuple):
                # Old format with tuples
                for h in history:
                    if isinstance(h, tuple) and len(h) == 2:
                        flat_history.extend(h)
            elif isinstance(history[0], str):
                # Already flat list
                flat_history = history
    
    # Get response with context from previous exchanges
    response = query(message, flat_history)
    return response

# Create a custom chat interface using Blocks instead of ChatInterface
# to ensure compatibility with older Gradio versions
with gr.Blocks(css="""
    footer {visibility: hidden}
    .gradio-container {max-width: 850px; margin: auto;}
    .message-bubble.user {background-color: #f9f9f9; border-radius: 10px; padding: 10px;}
    .message-bubble.bot {background-color: #f0f7ff; border-radius: 10px; padding: 10px;}
""") as demo:
    with gr.Row():
        gr.Markdown("""
        # 🧠 Advanced Educational AI Assistant
        
        Powered by cutting-edge small language models optimized for the free tier.
        This assistant provides high-quality educational answers while staying within resource constraints.
        """)
    
    # Create a custom chat interface using Chat and other components
    # Use tuples format instead of messages to avoid compatibility issues
    chatbot = gr.Chatbot(
        label="Conversation", 
        elem_classes="chatbot",
        height=500
    )
    
    msg = gr.Textbox(
        placeholder="Ask me any educational question...",
        show_label=False,
        container=False
    )
    
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear")
    
    # Add examples
    gr.Examples(
        examples=[
            "Explain quantum entanglement in terms a high school student could understand",
            "What are three different approaches to solving the traveling salesman problem?",
            "How does CRISPR gene editing technology work?",
            "What are the most effective teaching methods according to recent research?",
            "Write a short poem about mathematics that includes metaphors related to exploration"
        ],
        inputs=msg
    )
    
    # Define the chat functionality
    def respond(message, chat_history):
        if message.strip() == "":
            return chat_history
        
        # Show typing indicator by appending user message immediately
        chat_history.append((message, "..."))
        yield chat_history
        
        # Get AI response
        bot_message = chat_with_model(message, chat_history[:-1])
        
        # Update with actual response
        chat_history[-1] = (message, bot_message)
        yield chat_history
    
    # Connect components
    msg.submit(respond, [msg, chatbot], [chatbot], queue=True).then(
        lambda: "", None, [msg], queue=False
    )
    
    submit_btn.click(respond, [msg, chatbot], [chatbot], queue=True).then(
        lambda: "", None, [msg], queue=False
    )
    
    clear_btn.click(lambda: [], None, [chatbot], queue=False)
    
    with gr.Accordion("Advanced Settings & Information", open=False):
        with gr.Tabs():
            with gr.TabItem("Model Information"):
                gr.Markdown("""
                ### Model: Microsoft Phi-3-mini-4k-instruct
                - 2.8 billion parameters (under 10GB)
                - Highly optimized for instruction following
                - Performs at the level of much larger models on many tasks
                - Context window of 4,096 tokens for longer conversations
                
                ### Performance Optimizations
                - Response caching system to reduce API calls
                - Smart retry mechanism with exponential backoff
                - Concurrent request limiting to prevent quota issues
                """)
            
            with gr.TabItem("Embedding"):
                gr.Markdown("### Embed this chatbot in your website")
                space_url = os.getenv('SPACE_URL', 'YOUR_SPACE_URL_HERE')
                iframe_code = f"""
                <iframe
                    src="{space_url}"
                    width="100%"
                    height="700px"
                    style="border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
                    allow="microphone"
                ></iframe>
                """
                gr.Code(value=iframe_code, language="html")
            
            with gr.TabItem("System Tools"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Cache Management")
                        cache_files = len(os.listdir(CACHE_DIR)) if os.path.exists(CACHE_DIR) else 0
                        cache_size_mb = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR)) / (1024*1024) if os.path.exists(CACHE_DIR) else 0
                        cache_info = gr.Markdown(f"Cache entries: {cache_files} files ({cache_size_mb:.2f} MB)")
                        
                        clear_cache_btn = gr.Button("🧹 Clear Response Cache")
                        
                        def clear_cache():
                            if os.path.exists(CACHE_DIR):
                                deleted = 0
                                for file in os.listdir(CACHE_DIR):
                                    try:
                                        os.remove(os.path.join(CACHE_DIR, file))
                                        deleted += 1
                                    except Exception as e:
                                        logger.error(f"Failed to delete {file}: {e}")
                            return f"Cache cleared: {deleted} files removed"
                        
                        clear_cache_btn.click(clear_cache, outputs=[cache_info])
                    
                    with gr.Column():
                        gr.Markdown("### Model Settings")
                        model_dropdown = gr.Dropdown(
                            choices=[
                                "microsoft/Phi-3-mini-4k-instruct (default, best overall)",
                                "mistralai/Mistral-7B-Instruct-v0.2 (larger, higher quality)",
                                "google/gemma-2b-it (alternative small model)",
                                "TinyLlama/TinyLlama-1.1B-Chat-v1.0 (smallest option)"
                            ],
                            value="microsoft/Phi-3-mini-4k-instruct (default, best overall)",
                            label="Model Selection (requires restart)"
                        )
                        gr.Markdown("Note: Model changes require application restart")

    # Add a sidebar with helpful tips
    with gr.Row():
        gr.Markdown("""
        ### 💡 Tips for Best Results
        - Be specific in your questions
        - For complex topics, break them into smaller questions
        - For code examples, specify the programming language
        - Try the example questions to see the capabilities
        """)

# Get port from environment variable for Render compatibility
port = int(os.environ.get("PORT", 7860))

# Log startup information
logger.info(f"Starting application on port {port}")
logger.info(f"HF_TOKEN configured: {'Yes' if get_token() else 'No'}")

if __name__ == "__main__":
    try:
        # For Render deployment - bind to 0.0.0.0 with the PORT env variable
        demo.launch(server_name="0.0.0.0", server_port=port, share=False)
    except Exception as e:
        logger.error(f"Failed to launch app: {e}")
        print(f"Error launching application: {e}")
