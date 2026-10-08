from groq import Groq

client = Groq()
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Say hello in one word"}]
)
print(response.choices[0].message.content)