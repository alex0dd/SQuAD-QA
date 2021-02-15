import torch

from models.utils import SpanExtractor

def extract_answer(paragraph_tokens, start_idx, end_idx):
    answer_tokens = []
    if start_idx >= len(paragraph_tokens):
        # out of bounds
        print("Might fail", paragraph_tokens, "len",len(paragraph_tokens) ,"start",start_idx)
    elif start_idx == end_idx:
        answer_tokens.append(paragraph_tokens[start_idx])
    else:
        for i in range(min(end_idx-start_idx, len(paragraph_tokens)-start_idx)):
            answer_tokens.append(paragraph_tokens[start_idx+i])
    return " ".join(answer_tokens)


def build_evaluation_dict(model, dataloader, paragraphs_mapper, device):
    # Build the evaluation dict
    eval_dict = {}
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            answer_spans_start = batch["y_gt"][:, 0]
            answer_spans_end = batch["y_gt"][:, 1]
            paragraph_id = batch["paragraph_id"]
            question_id = batch["question_id"]
            # Place to right device
            answer_spans_start = answer_spans_start.to(device)
            answer_spans_end = answer_spans_end.to(device)
            # Run forward pass
            pred_answer_start_scores, pred_answer_end_scores = model(batch)
            # Get span indexes
            pred_span_start_idxs, pred_span_end_idxs = SpanExtractor.extract_most_probable(pred_answer_start_scores, pred_answer_end_scores)
            # extract answer texts from paragraphs
            for sample_idx in range(len(paragraph_id)):
                paragraph_sample_id = paragraph_id[sample_idx]
                question_sample_id = question_id[sample_idx]
                pred_span_start_sample = pred_span_start_idxs[sample_idx]
                pred_span_end_sample = pred_span_end_idxs[sample_idx]
                pred_answer_text = extract_answer(paragraphs_mapper[paragraph_sample_id],
                                                  pred_span_start_sample,
                                                  pred_span_end_sample)
                # add new (question_id, pred_answer_text) to the eval dict:
                eval_dict[question_sample_id] = pred_answer_text
            
    return eval_dict