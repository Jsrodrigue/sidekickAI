from pydantic import BaseModel, Field



class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Retroalimentación sobre la respuesta")
    success_criteria_met: bool = Field(description="Si se cumplieron los criterios")
    user_input_needed: bool = Field(description="Si se necesita más input del usuario")
    confidence: float = Field(description="Confianza en la evaluación (0-1)", ge=0, le=1)

