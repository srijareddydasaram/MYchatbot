import ollama

print("AI Chatbot")
print("Type 'exit' to stop.\n")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        print("Chat ended.")
        break

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "user", "content": user}
        ]
    )

    print("AI:", response["message"]["content"])