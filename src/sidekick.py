import os
import uuid
from typing import List, Any, Optional, Dict, Annotated
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, PythonLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv(override=True)


class SidekickState(BaseModel):

    messages: Annotated[List[Any], add_messages] = Field(default_factory=list)

    # REQUIRED FOR LANGGRAPH CHECKPOINTER
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    success_criteria: Optional[str] = None
    task_metadata: Dict[str, Any] = Field(default_factory=dict)

    current_directory: Optional[str] = None
    indexed_directories: List[str] = Field(default_factory=list)

    evaluation_history: List[Dict[str, Any]] = Field(default_factory=list)
    criteria_met: bool = False
    needs_user_input: bool = False

    class Config:
        arbitrary_types_allowed = True


# ------------------------
# Schemas estructurados
# ------------------------
class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Retroalimentación sobre la respuesta")
    success_criteria_met: bool = Field(description="Si se cumplieron los criterios")
    user_input_needed: bool = Field(description="Si se necesita más input del usuario")
    confidence: float = Field(description="Confianza en la evaluación (0-1)", ge=0, le=1)

# ------------------------
# Sidekick
# ------------------------
class Sidekick:
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.sidekick_id = str(uuid.uuid4())
        
        # LLMs
        self.worker_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)
        self.evaluator_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)
        
        # Embeddings and storage
        self.emb = OpenAIEmbeddings(openai_api_key=api_key)
        self.retriever_registry: Dict[str, Any] = {}
        
        # Memory
        self.memory = MemorySaver()
        self.graph = None
        
        # Tools to be created in setup
        self.tools = []

    async def setup(self):
        """Inicializa el grafo y las herramientas"""
        # Crear tool con closure para acceder al registry
        @tool
        def search_documents(query: str, k: int = 5) -> str:
            """
            Busca información en los documentos del directorio activo.
            
            Args:
                query: La consulta de búsqueda
                k: Número de documentos a retornar
            """
            return self._rag_query(query, k)
        
        self.tools = [search_documents]
        self.worker_llm_with_tools = self.worker_llm.bind_tools(self.tools)
        self.evaluator_llm_with_output = self.evaluator_llm.with_structured_output(EvaluatorOutput)
        
        await self.build_graph()

    # ------------------------
    # RAG: Indexation
    # ------------------------
    def load_and_register_directory(self, directory: str) -> str:
        """Indexa un directorio (txt, md, py, pdf) excluyendo .venv, venv y __pycache__."""

        if not directory or not os.path.exists(directory):
            return f"❌ Invalid directory: {directory}"

        excluded_dirs = {".venv", "venv", "__pycache__"}

        def is_excluded(path: str) -> bool:
            """Verifica si la ruta contiene directorios excluidos."""
            norm = path.replace("\\", "/")
            parts = norm.split("/")
            return any(ex in parts for ex in excluded_dirs)

        docs = []

        # Loaders normales (txt, md, py)
        loaders = [
            DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
            DirectoryLoader(directory, glob="**/*.md", loader_cls=TextLoader, show_progress=True),
            DirectoryLoader(directory, glob="**/*.py", loader_cls=PythonLoader, show_progress=True),
        ]

        for loader in loaders:
            try:
                loaded = loader.load()
                for doc in loaded:
                    src = doc.metadata.get("source", "")
                    if not is_excluded(src):
                        docs.append(doc)
            except Exception as e:
                print(f"⚠️ Error cargando con {loader.__class__.__name__}: {e}")

        # PDFs: carga manual
        import glob as glob_module
        pdf_pattern = os.path.join(directory, "**/*.pdf")
        pdf_files = glob_module.glob(pdf_pattern, recursive=True)

        for pdf_path in pdf_files:
            if is_excluded(pdf_path):
                continue
            
            try:
                loader = PyPDFLoader(pdf_path)
                pdf_docs = loader.load()
                docs.extend(pdf_docs)
                print(f"✅ PDF cargado: {os.path.basename(pdf_path)}")
            except Exception as e:
                print(f"⚠️ Error cargando PDF {pdf_path}: {e}")

        if not docs:
            return "❌ No se encontraron documentos legibles"

        # Metadata extra
        for doc in docs:
            src = doc.metadata.get("source", "")
            doc.metadata["file_name"] = os.path.basename(src)
            doc.metadata["file_path"] = src
            doc.metadata["indexed_at"] = datetime.now().isoformat()

        # Chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=20,
            length_function=len,
        )
        chunks = splitter.split_documents(docs)

        # Vectorstore persistente
        persist_dir = f"vector_db/{os.path.basename(directory)}_{uuid.uuid4().hex[:8]}"
        os.makedirs(persist_dir, exist_ok=True)

        vectorstore = Chroma.from_documents(
            chunks,
            self.emb,
            persist_directory=persist_dir
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        self.retriever_registry[directory] = retriever

        return f"✅ Indexados {len(chunks)} fragmentos de {len(docs)} documentos en '{directory}'"

    # ------------------------
    # RAG: Búsqueda (método interno)
    # ------------------------
    def _rag_query(self, query: str, k: int = 5) -> str:
        """Método interno para buscar en el directorio activo"""
        # Obtener directorio del contexto (thread_id en producción)
        active_dir = list(self.retriever_registry.keys())[0] if self.retriever_registry else None
        
        if not active_dir:
            return "❌ No hay directorio indexado. Usa load_and_register_directory() primero."
        
        retriever = self.retriever_registry[active_dir]
        
        try:
            docs = retriever.invoke(query)[:k]
            
            if not docs:
                return f"❌ No se encontraron documentos relevantes para: '{query}'"
            
            # Formatear resultados
            results = []
            for i, doc in enumerate(docs, 1):
                fname = doc.metadata.get("file_name", "desconocido")
                content = doc.page_content[:500]  # Limitar tamaño
                results.append(f"📄 **Documento {i}** ({fname}):\n{content}\n")
            
            return "\n---\n".join(results)
            
        except Exception as e:
            return f"❌ Error en búsqueda: {str(e)}"

    # ------------------------
    # Nodos del grafo
    # ------------------------
    def worker(self, state: SidekickState) -> dict:
        """Nodo que procesa la solicitud del usuario"""
        system_msg = SystemMessage(content=f"""Eres un asistente útil con acceso a herramientas.

**Criterios de éxito:** {state.success_criteria or "Responder de forma clara y precisa"}

**Herramientas disponibles:**
- search_documents: Busca información en documentos indexados

**Hora actual:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

Usa las herramientas cuando necesites información externa. Sé conciso y directo.
""")
        
        messages = [system_msg] + state.messages
        response = self.worker_llm_with_tools.invoke(messages)
        
        return {"messages": [response]}

    def should_continue(self, state: SidekickState) -> str:
        """Decide si llamar tools o ir al evaluator"""
        last_message = state.messages[-1]
        
        # Si el mensaje tiene tool_calls, ir a tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Si no, evaluar
        return "evaluator"

    def evaluator(self, state: SidekickState) -> dict:
        """Evalúa si se cumplieron los criterios de éxito"""
        # Construir contexto de conversación
        conversation = "\n".join([
            f"{'👤 Usuario' if isinstance(m, HumanMessage) else '🤖 Asistente'}: {getattr(m, 'content', '')}"
            for m in state.messages[-6:]  # Últimos 6 mensajes
        ])
        
        last_response = state.messages[-1].content if state.messages else ""
        
        eval_prompt = f"""Evalúa la siguiente conversación:

**Conversación:**
{conversation}

**Última respuesta del asistente:**
{last_response}

**Criterios de éxito:**
{state.success_criteria or "Responder de forma clara y precisa"}

Evalúa si:
1. La respuesta cumple los criterios
2. Es necesaria más información del usuario
3. La respuesta es completa y útil
"""
        
        eval_result = self.evaluator_llm_with_output.invoke([
            SystemMessage(content="Eres un evaluador objetivo de respuestas de IA."),
            HumanMessage(content=eval_prompt)
        ])
        
        # Guardar evaluación
        eval_dict = eval_result.model_dump()
        eval_dict["timestamp"] = datetime.now().isoformat()
        
        return {
            "evaluation_history": [eval_dict],
            "criteria_met": eval_result.success_criteria_met,
            "needs_user_input": eval_result.user_input_needed,
            "messages": [AIMessage(content=f"💭 Evaluación: {eval_result.feedback}")]
        }

    def should_end(self, state: SidekickState) -> str:
        """Decide si terminar o continuar trabajando"""
        if state.criteria_met or state.needs_user_input:
            return "end"
        return "worker"

    # ------------------------
    # Construcción del grafo
    # ------------------------
    async def build_graph(self):
        """Construye el grafo de ejecución"""
        graph_builder = StateGraph(SidekickState)
        
        # Nodos
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)
        
        # Edges
        graph_builder.add_edge(START, "worker")
        
        graph_builder.add_conditional_edges(
            "worker",
            self.should_continue,
            {"tools": "tools", "evaluator": "evaluator"}
        )
        
        graph_builder.add_edge("tools", "worker")
        
        graph_builder.add_conditional_edges(
            "evaluator",
            self.should_end,
            {"worker": "worker", "end": END}
        )
        
        self.graph = graph_builder.compile(checkpointer=self.memory)
        return self.graph

    # ------------------------
    # Ejecución
    # ------------------------
    async def run(self, user_input: str, thread_id: Optional[str] = None) -> str:
        """Ejecuta el sidekick con un input del usuario"""
        if not self.graph:
            await self.setup()
        
        thread_id = self.sidekick_id
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = SidekickState(
            messages=[HumanMessage(content=user_input)],
            success_criteria="Responder la pregunta del usuario de forma completa"
        )
        
        result = await self.graph.ainvoke(initial_state, config)
        
        # Retornar última respuesta del asistente
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not msg.content.startswith("💭"):
                return msg.content
        
        return "No se generó respuesta"

    def cleanup(self):
        """Limpia recursos"""
        self.retriever_registry.clear()
        print("🧹 Recursos limpiados")


