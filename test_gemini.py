import os
import sys

# Add the project root to path so we can import backend modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# FIX 1: Import config to ensure load_dotenv() is called
from backend.config import settings

import google.generativeai as genai

print("=" * 70)
print("TEST_GEMINI.PY - DEBUG OUTPUT")
print("=" * 70)

# Test 1: Using os.getenv() directly (before config import)
print("\n1. BEFORE importing settings:")
print(f"   API KEY FOUND (os.getenv): {bool(os.getenv('GEMINI_API_KEY'))}")
print(f"   MODEL (os.getenv): {os.getenv('GEMINI_MODEL')}")

# Test 2: Using settings object (after config import)
print("\n2. AFTER importing settings from config:")
print(f"   API KEY FOUND (settings object): {bool(settings.gemini_api_key)}")
print(f"   API KEY value (first 20 chars): {settings.gemini_api_key[:20] if settings.gemini_api_key else 'NOT SET'}")
print(f"   MODEL (settings object): {settings.gemini_model}")

# Test 3: Attempt to use Gemini API
print("\n3. ATTEMPTING GEMINI API CALL:")
if settings.gemini_api_key:
    try:
        genai.configure(api_key=settings.gemini_api_key)
        print(f"   genai.configure() - SUCCESS")
        print(f"   Creating model with: {settings.gemini_model}")
        model = genai.GenerativeModel(settings.gemini_model)
        print(f"   GenerativeModel created - SUCCESS")
        response = model.generate_content("Say hello")
        print(f"   generate_content() - SUCCESS")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ERROR: {e}")
else:
    print("   SKIPPED - API KEY NOT SET")

print("=" * 70)