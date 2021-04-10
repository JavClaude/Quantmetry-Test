from typing import List
from pydantic import BaseModel

class InModel(BaseModel):
    age: int
    exp: int
    salaire: int
    diplome: str
    specialite: str
    note: int
    dispo: str

class OutModel(BaseModel):
    scores: List[float]