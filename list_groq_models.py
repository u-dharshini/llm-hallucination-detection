from groq import Groq
from backend.app.config import settings
client = Groq(api_key=settings.GROQ_API_KEY)
models = client.models.list()
for m in models.data:
    print(m.id)
