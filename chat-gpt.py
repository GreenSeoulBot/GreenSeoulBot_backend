import openai
import os

OPENAI_API_KEY = os.getenv("API-KEY")
openai.api_key = OPENAI_API_KEY

