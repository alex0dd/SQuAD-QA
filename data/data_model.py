from dataclasses import dataclass
from typing import List

@dataclass
class Answer:
    answer_start: int
    text: str

@dataclass
class Question:
    question: str
    id: str
    answers: List[Answer]

@dataclass
class Paragraph:
    context: str
    questions: List[Question]

@dataclass
class Document:
    title: str
    paragraphs: List[Paragraph]
