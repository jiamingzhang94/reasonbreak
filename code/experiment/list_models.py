#!/usr/bin/env python3
"""
Simple script to list available Google Generative AI models
Usage: python list_models.py
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import google.generativeai as genai
    print("Google Generative AI SDK is available")
except ImportError:
    print("Error: google-generativeai package not available")
    print("Please install it with: pip install google-generativeai")
    exit(1)

def list_google_models():
    """List available Google Generative AI models"""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment")
        print("Please set GOOGLE_API_KEY in your .env file")
        return
    
    try:
        genai.configure(api_key=api_key)
        
        print("\nAvailable Google Generative AI models:")
        print("=" * 50)
        
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"✓ {model.name}")
                if hasattr(model, 'description') and model.description:
                    print(f"  Description: {model.description}")
                if hasattr(model, 'input_token_limit') and model.input_token_limit:
                    print(f"  Input token limit: {model.input_token_limit:,}")
                if hasattr(model, 'output_token_limit') and model.output_token_limit:
                    print(f"  Output token limit: {model.output_token_limit:,}")
                print()
                
    except Exception as e:
        print(f"Failed to list Google models: {e}")
        print("\nPossible solutions:")
        print("1. Check your GOOGLE_API_KEY is correct")
        print("2. Ensure you have internet connectivity")
        print("3. Verify your API key has proper permissions")

def test_model(model_name):
    """Test a specific model with a simple text prompt"""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found")
        return
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        print(f"\nTesting model: {model_name}")
        print("-" * 30)
        
        response = model.generate_content("Hello, can you confirm you're working?")
        print(f"Response: {response.text}")
        print("✓ Model is working correctly!")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")

if __name__ == "__main__":
    print("Google Generative AI Model Explorer")
    print("=" * 40)
    
    list_google_models()
    
    # Test the default model
    test_model("gemini-1.5-pro")
    
    print("\nRecommended models for your use case:")
    print("- gemini-1.5-pro: Best for complex reasoning and long context")
    print("- gemini-1.5-flash: Faster, good for simpler tasks")
    print("- gemini-2.0-flash-exp: Latest experimental version (if available)") 