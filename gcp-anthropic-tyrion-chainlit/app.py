import os
import anthropic
import chainlit as cl

# Initialize the Asynchronus Anthropic class
async_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@cl.on_chat_start
async def start_chat() -> None:
    """
    Initialize the chat session.

    This function is called when a new chat session starts. It sends an initial
    message and sets up the prompt history in the user session.
    """
    await cl.Message(content="Never forget who you are, for surely the world won’t. Make it your strength. ").send()
    cl.user_session.set(
        "prompt_history",
        "",
    )

async def call_claude(query: str) -> None:
    """
    Call the Claude AI model and stream the response.

    This function sends a query to the Claude AI model, retrieves the response,
    and streams it back to the user.

    Args:
        query (str): The user's input query.
    """
    prompt_history = cl.user_session.get("prompt_history")

    prompt = f"{prompt_history}{anthropic.HUMAN_PROMPT}{query}{anthropic.AI_PROMPT}"

    system_prompt = """You are to act as Tyrion Lannister from Game of Thrones. Embody his wit, intelligence, cynicism, and complex personality in your responses. 
        Draw upon your knowledge of Westerosi politics, history, and culture. Use Tyrion's characteristic sarcasm, wordplay, and references to drinking when appropriate. 
        Remember that Tyrion is well-read, politically savvy, and often shows compassion for outcasts despite his cynical worldview.

        Key traits to incorporate:
        1. Sharp intellect and wit
        2. Sarcastic humor, often self-deprecating
        3. Love of wine and books
        4. Complex family relationships, especially with Lannisters
        5. Experience in politics and warfare
        6. Sympathy for "cripples, bastards, and broken things"
        7. Use of clever wordplay and eloquent speech
        8. Cynical worldview balanced with moments of idealism

        Respond to prompts as Tyrion would, maintaining his voice and perspective throughout. Adapt your language and references to fit the world of Tyrion's experiences.
        Keep the conversation simple unless the user asks for.
    """

    msg = cl.Message(content="", author="Tyrion")

    stream = await async_client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=512,
    temperature=0.8,
    stop_sequences=[anthropic.HUMAN_PROMPT],
    stream=True,
    system=system_prompt,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

    async for data in stream:
        if data.type == "content_block_delta":
            token = data.delta.text
            await msg.stream_token(token)

    await msg.send()
    cl.user_session.set("prompt_history", prompt + msg.content)


@cl.on_message
async def chat(message: cl.Message) -> None:
    """
    Handle incoming chat messages.

    This function is called whenever a new message is received in the chat.

    Args:
        message (cl.Message): The incoming chat message.
    """
    await call_claude(message.content)