from backend.app.config import settings
print(repr(settings.LLM_PROVIDER))
print(repr(settings.LLM_MODEL))
print(repr(settings.GROQ_API_KEY[:10]))
