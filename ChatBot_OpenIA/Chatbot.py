import openai

openai.api_key = "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"

messages = []
system_msg = input("Que tipo de chatbot te gustaria crear?\n")
messages.append({"role": "system", "content": system_msg})

print("Tu nuevo asistente esta lsito!")
while input != "quit()":
    message = input()
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")
