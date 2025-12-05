from langchain_core.messages import AIMessage, HumanMessage

from src.core.sidekick import Sidekick


class SidekickService:
    def __init__(self, sidekick: Sidekick):
        self.sidekick = sidekick

    async def send_message(
        self, prompt: str, state, username: str, folder: str | None = None
    ):
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        thread_id = f"{username}:{folder}" if folder else username

        state.messages.append(HumanMessage(content=prompt))

        assistant_reply = await self.sidekick.run(
            prompt, thread_id=thread_id, folder=folder
        )

        state.messages.append(AIMessage(content=assistant_reply))

        return state
