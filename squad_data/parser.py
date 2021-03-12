import json
import os

from typing import List

from . import Answer, Question, Paragraph, Document


class SquadFileParser:
    '''
    Parser for SQuAD format datasets.

    Args:
        path (str): path of SQuAD dataset json file
    '''

    def __init__(self, path: str):
        self.path = path
        self.exists = os.path.exists(self.path)

    def __parse_answer(self, answer) -> Answer:
        return Answer(answer_start=answer['answer_start'], text=answer['text'])

    def __parse_question(self, question) -> Question:
        answers: List[Answer] = []
        if "answers" in question:
        	for answer in question["answers"]:
        		answers.append(self.__parse_answer(answer))
        		
        return Question(
                question=question["question"], 
                id=question["id"],
                answers=answers
            )

    def __parse_paragraph(self, paragraph) -> Paragraph:
        questions: List[Question] = []
        for question in paragraph["qas"]:
            questions.append(self.__parse_question(question))
        return Paragraph(
                context=paragraph["context"],
                questions=questions
            )

    def __parse_document(self, document) -> Document:
        paragraphs: List[Paragraph] = []
        for paragraph in document["paragraphs"]:
            paragraphs.append(self.__parse_paragraph(paragraph))
        return Document(title=document["title"], paragraphs=paragraphs)

    def parse_documents(self) -> List[Document]:
        documents: List[Document] = []
        if self.exists:
            with open(self.path, 'r') as f:
                contents = json.load(f)
                data = contents["data"]
                for document in data:
                    documents.append(self.__parse_document(document))
        return documents
