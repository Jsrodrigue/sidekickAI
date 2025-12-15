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
        enabled_tools: list[str] | None = None,
    ):
        """
        Send a user message through Sidekick and update the session state.

        - state.messages contiene TODO el historial del usuario.
        - Para dar memoria sin mucha latencia, pasamos al grafo solo
          una ventana de los últimos N mensajes.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Append user message to state history (persisted in your DB)
        state.messages.append(HumanMessage(content=prompt))

        # ---- Ventana de historial para el grafo ----
        MAX_HISTORY_MESSAGES = 12  # p.ej. 6 turnos user+assistant
        history_slice = state.messages[-MAX_HISTORY_MESSAGES:]

        # Run Sidekick pipeline (LangGraph + tools + RAG) with recent history
        assistant_reply = await self.sidekick.run(
            user_input=prompt,
            folder=folder,
            top_k=top_k,
            enabled_tools=enabled_tools,
            history=history_slice,  # 👉 solo los últimos N mensajes
        )

        # Append assistant reply to state (historial persistente)
        state.messages.append(AIMessage(content=assistant_reply))

        # Return updated state so UIHandlers can persist it
        return state
