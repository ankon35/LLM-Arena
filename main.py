from chatbot import ChatBotSystem

def main():
    print("\n=== Multi-Agent Debate System ===\n")
    print("1. OpenAI")
    print("2. Gemini")
    print("3. Claude")
    print("4. Debate Mode (Multi-Agent AI vs AI)\n")

    choice = input("Select option (1-4): ").strip()

    bot = ChatBotSystem(choice)

    print("\nType 'exit' to stop\n")

    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            break

        response = bot.get_response(query)
        print("\nAI:\n", response)


if __name__ == "__main__":
    main()