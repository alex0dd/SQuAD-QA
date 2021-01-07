import pandas as pd

from . import Document
from typing import Tuple, List, Dict

def build_mappers_and_dataframe(documents_list: List[Document]) -> Tuple[Dict[str, str], Dict[str, str], pd.DataFrame]:
    """
    Given a list of SQuAD Document objects, returns mappers to transform from
    paragraph id to paragraph text, question id to question text, and
    a dataframe containing paragraph id, question id and answer details.

    Args:
        documents_list (List[Document]): list of parsed SQuAD document objects.

    Returns:
        paragraphs_mapper: mapper from paragraph id to paragraph text
        questions_mapper: mapper from question id to question text
        dataframe: Pandas dataframe with the following schema
            (paragraph_id, question_id, answer_id, answer_start, answer_text)
    """

    # type for np array: np.ndarray
    # given a paragraph id, maps the paragraph to its text or embeddings (or both)
    paragraphs_mapper = {}
    # given a question id, maps the question to its text or embeddings (or both)
    questions_mapper = {}
    # dataframe
    dataframe_list = []
    for doc_idx, document in enumerate(documents_list):
        # for each paragraph
        for par_idx, paragraph in enumerate(document.paragraphs):
            par_id = "{}_{}".format(doc_idx, par_idx)
            paragraphs_mapper[par_id] = paragraph.context.strip()
            # for each question
            for question in paragraph.questions:
                question_id = question.id
                questions_mapper[question_id] = question.question.strip()
                for answer_id, answer in enumerate(question.answers):
                    # build dataframe entry
                    dataframe_list.append({
                        "paragraph_id": par_id,
                        "question_id": question_id,
                        "answer_id": answer_id,
                        "answer_start": answer.answer_start,
                        "answer_text": answer.text.strip()
                    })
    return paragraphs_mapper, questions_mapper, pd.DataFrame(dataframe_list)
