from langchain_core.messages import AIMessage, HumanMessage

from src.core.sidekick import Sidekick


class SidekickService:
    def __init__(self, sidekick: Sidekick):
        self.sidekick = sidekick

    async def send_message(
        self,
        prompt: str,
        state,
        username: str,
        folder: str | None = None,
        top_k: int | None = None,
    ):
        """
        Send a user message through Sidekick and update the session state.

        Parameters
        ----------
        prompt : str
            User's message.
        state :
            Session state object, expected to have a `.messages` list.
        username : str
            Current logged-in username (used for thread_id).
        folder : str | None
            Active folder for RAG (used to select the retriever).
        top_k : int | None
            If provided, controls how many documents are retrieved
            per query (RAG). Passed to Sidekick.run().
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Thread separation: one logical conversation per (user, folder)
        thread_id = f"{username}:{folder}" if folder else username

        # Append user message to state
        state.messages.append(HumanMessage(content=prompt))

        # Run Sidekick pipeline (LangGraph + tools + RAG)
        assistant_reply = await self.sidekick.run(
            user_input=prompt,
            thread_id=thread_id,
            folder=folder,
            top_k=top_k,
        )

        # Append assistant reply to state
        state.messages.append(AIMessage(content=assistant_reply))

        # Return updated state so UIHandlers can persist it
        return state
