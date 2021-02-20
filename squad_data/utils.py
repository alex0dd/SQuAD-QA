import pandas as pd

from . import Document
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import Union, Tuple, List, Dict, Any

def index_of_first(lst, pred):
    for i, v in enumerate(lst):
        if pred(v):
            return i
    return None

def split_paragraph_if_needed(paragraph, question, answer_span, tokenizer, tokenizer_fn):
    """
    Attempts to tokenize a paragraph and question together, if too long
    because of tokenizer's max length, then will split the paragraph into
    multiple slices.
    
    Returns a list of paragraph slices with answer span, such that:
        - a paragraph slice with no answer will have answer mapped to (CLS, CLS)
        - a paragraph slice with answer will be mapped to the index of answer.
    """
    tokenized_input_pair = tokenizer_fn(question, paragraph)
    # outputs
    paragraph_splits = []
    answer_spans = []
    # get answer end char idx
    ans_start = answer_span[0]
    ans_end = answer_span[1]
    """
    1) Find index of context segments in the tokenized example
    2) Within the context segments (start from context_segment_idx), 
       find the token corresponding to span of answer: start and end.
    """
    for offset_idx, offset in enumerate(tokenized_input_pair.offset_mapping):
        # get sequence ids
        sequence_ids = tokenized_input_pair.sequence_ids(offset_idx)
        # find start index of context segment
        context_segment_idx = sequence_ids.index(1)
        # TODO(Alex): ADD QUICK FIX WITH n_special_tokens (but it's not a proper solution)
        span_start_offset_idx = index_of_first(
            tokenized_input_pair.offset_mapping[offset_idx][context_segment_idx:], 
            lambda span: span[0] <= ans_start <= span[1]
        )
        span_end_offset_idx = index_of_first(
            tokenized_input_pair.offset_mapping[offset_idx][context_segment_idx:], 
            lambda span: span[0] <= ans_end <= span[1]
        )
        # Decode split into a string
        decoded_split = tokenizer.decode(tokenized_input_pair.input_ids[offset_idx][context_segment_idx:], skip_special_tokens=True)
        # 
        paragraph_splits.append(decoded_split)
        if span_start_offset_idx is not None and span_end_offset_idx is not None:
            # If answer span is fully in current slice
            # add segment idx offset
            span_start_offset_idx += context_segment_idx
            span_end_offset_idx += context_segment_idx + 1 # the plus 1 is needed for correct slicing
            answer_spans.append((span_start_offset_idx, span_end_offset_idx))
        elif span_start_offset_idx is None and span_end_offset_idx is None:
            # If span not in this slice, but in another slice
            # map answer to (CLS, CLS)
            cls_idx = tokenized_input_pair.input_ids[offset_idx].index(tokenizer.cls_token_id)
            # NOTE(Alex): although I think it's always 0
            answer_spans.append((cls_idx, cls_idx))
        else:
            # span spans along multiple slices -> throw the sample away 
            # (should be only like 4 samples across the whole dataset)
            # Discard sample
            pass
    
    return (paragraph_splits, answer_spans)

def build_mappers_and_dataframe_bert(
    tokenizer,
    tokenizer_fn,
    documents_list: List[Document], 
    limit_answers: int = -1
    ) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Given a list of SQuAD Document objects, returns mapper to transform from
    paragraph id to paragraph text and a dataframe containing paragraph id, 
    question id, text and answer details. The function also ensures that
    returned paragraph text won't exceed maximum length required by BERT, by
    splitting long paragraphs into multiple parts.
    Args:
        tokenizer: Huggingface tokenizer
        tokenizer_fn: helper encoder function for tokenizer
        documents_list (List[Document]): list of parsed SQuAD document objects.
        limit_answers (int): limit number of returned answers per question
            to this amount (-1 to return all the available answers).

    Returns:
        paragraphs_mapper: mapper from paragraph id to paragraph text
        dataframe: Pandas dataframe with the following schema
            (paragraph_id, question_id, question_text, answer_id, answer_start, answer_text)
    """

    # type for np array: np.ndarray
    # given a paragraph id, maps the paragraph to its text
    split_paragraphs_mapper = {}
    # dataframe
    dataframe_list = []
    for doc_idx, document in enumerate(documents_list):
        # for each paragraph
        for par_idx, paragraph in enumerate(document.paragraphs):
            par_text = paragraph.context.strip()
            # for each question
            for question in paragraph.questions:
                question_id = question.id
                question_text = question.question.strip()
                # take only "limit_answers" answers for every question.
                answer_range = len(question.answers) if limit_answers == -1 else limit_answers
                for answer_id, answer in enumerate(question.answers[:answer_range]):
                    # NOTE: in training set, there's only one answer per question.
                    answer_text = answer.text.strip()
                    # get span
                    answer_start = answer.answer_start
                    answer_end = answer.answer_start + len(answer_text)

                    par_splits, split_answer_spans = split_paragraph_if_needed(
                        par_text, 
                        question_text, 
                        (answer_start, answer_end),
                        tokenizer,
                        tokenizer_fn
                    )
                    
                    pair_overflows = len(par_splits) > 1
                    
                    for split_idx, (split_text, split_ans_span) in enumerate(zip(par_splits, split_answer_spans)):
                        """
                        NOTE(Alex): 
                        Since in tokenization phase we also use question, our ID depends on question too
                        For example if for question1, the pair <question1, par> goes above the limit,
                        but for <question2, par> it does not, then we'll still need to keep track of
                        different splits of par, depending on each question.
                        """
                        if pair_overflows:
                            split_par_id = "{}_{}_{}_{}".format(doc_idx, par_idx, question_id, split_idx)
                        else:
                            """
                            If no length overflow, then we don't need question_id or split_idx.
                            To optimize memory, we can map same splits to same id 
                            (some pairs <question, par> won't overflow anyway)
                            """
                            split_par_id = "{}_{}".format(doc_idx, par_idx)
                        split_paragraphs_mapper[split_par_id] = split_text
                        # build dataframe entry
                        dataframe_list.append({
                            "doc_id": doc_idx,
                            "paragraph_id": split_par_id,
                            "question_id": question_id,
                            "answer_id": answer_id,
                            "answer_start": answer_start,
                            "answer_text": answer_text,
                            "question_text": question_text,
                            "tokenizer_answer_start": split_ans_span[0],
                            "tokenizer_answer_end": split_ans_span[1],
                        })
    return split_paragraphs_mapper, pd.DataFrame(dataframe_list)

def build_mappers_and_dataframe(documents_list: List[Document], limit_answers: int = -1) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Given a list of SQuAD Document objects, returns mapper to transform from
    paragraph id to paragraph text and a dataframe containing paragraph id, 
    question id, text and answer details.
    Args:
        documents_list (List[Document]): list of parsed SQuAD document objects.
        limit_answers (int): limit number of returned answers per question
            to this amount (-1 to return all the available answers).

    Returns:
        paragraphs_mapper: mapper from paragraph id to paragraph text
        dataframe: Pandas dataframe with the following schema
            (paragraph_id, question_id, question_text, answer_id, answer_start, answer_text)
    """

    # type for np array: np.ndarray
    # given a paragraph id, maps the paragraph to its text or embeddings (or both)
    paragraphs_mapper = {}
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
                question_text = question.question.strip()
                # take only "limit_answers" answers for every question.
                answer_range = len(question.answers) if limit_answers == -1 else limit_answers
                for answer_id, answer in enumerate(question.answers[:answer_range]):
                    # build dataframe entry
                    dataframe_list.append({
                        "doc_id": doc_idx,
                        "paragraph_id": par_id,
                        "question_id": question_id,
                        "answer_id": answer_id,
                        "answer_start": answer.answer_start,
                        "answer_text": answer.text.strip(),
                        "question_text": question_text
                    })
    return paragraphs_mapper, pd.DataFrame(dataframe_list)

def get_spans_from_text(text: Union[str, List[str]]) -> List[Tuple[int,int]]:
    """
    Given a text string or a list of string tokens,
    return the list of spans, where each span represents
    the single word/token inside the text in the form of:
        ( start_position, start_position + len(token) )
    """
    current = 0
    spans = []

    if type(text) == str:
        tokens = text.split()
    else:
        # already tokenized
        tokens = text
        # restore string structure to use find on string
        text = " ".join(text)
    
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def add_paragraphs_spans(paragraphs_mapper: Dict[str, str]) -> Dict[str, Any]:
    """
    Extend the default paragraphs_mapper given by the build_mappers_and_dataframe
    function with an additional field containing the spanned paragraph text
 
    Args:
        paragraph_mapper (Dict[str, str]): mapper from paragraph id to paragraph text

    Returns:
        paragraph_spans_mapper (Dict[str, Any]): a mapper of the form
            {'paragraph id' : {'text', 'spans'} }
    """
    paragraphs_spans_mapper = {}

    for par_idx, paragraph in paragraphs_mapper.items():
        paragraphs_spans_mapper[par_idx] = {'text' : paragraph, 'spans' : get_spans_from_text(paragraph)}

    return paragraphs_spans_mapper
