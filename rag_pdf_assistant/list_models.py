import os
from dotenv import load_dotenv

load_dotenv()
try:
    from google import genai
    client = genai.Client()
    for m in client.models.list():
        print(m.name, m.supported_generation_methods)
except Exception as e:
    import google.generativeai as genai_old
    genai_old.configure(api_key=os.getenv("GEMINI_API_KEY"))
    for m in genai_old.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print(m.name)

