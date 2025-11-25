import uuid
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from typing import Annotated

class SidekickState(BaseModel):

    messages: Annotated[List[Any], add_messages] = Field(default_factory=list)

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

