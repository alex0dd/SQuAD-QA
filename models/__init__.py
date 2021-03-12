from .span_models import ParametricBertModelQA, ExtraParametricBertModelQA
# Parameters dictionary

def prepare_input_distilbert(inputs, device):
    model_input = {}
    model_input["input_ids"] = inputs["input_ids"].to(device)
    model_input["attention_mask"] = inputs["attention_mask"].to(device)
    return model_input

def prepare_input_albert(inputs, device):
    # for now we'll just copy distilbert since it works
    model_input = {}
    model_input["input_ids"] = inputs["input_ids"].to(device)
    model_input["attention_mask"] = inputs["attention_mask"].to(device)
    return model_input

possible_models_dict = {
    "distilbert" : {
        "model_url" : "distilbert-base-uncased",
        "tokenizer_url": "distilbert-base-uncased",
        "tokenizer_max_length": 384,
        "prepare_model_input_fn": prepare_input_distilbert,
        "span_model": ParametricBertModelQA,
        "train_params": {
            "epochs": 2,
            "initial_lr": 0.00003,
            "batch_size_train": 32,
            "batch_size_val": 32,
            "batch_size_test": 32,
            "weight_decay": 0.01,
            "dropout_rate": 0.1
        }
    },
    "albert": {
        "model_url": "albert-base-v2",
        "tokenizer_url": "albert-base-v2",
        "tokenizer_max_length": 384,
        "prepare_model_input_fn": prepare_input_albert,
        "span_model": ParametricBertModelQA,
        "train_params": {
            "epochs": 2,
            "initial_lr": 0.00003,
            "batch_size_train": 8,
            "batch_size_val": 8,
            "batch_size_test": 8,
            "weight_decay": 0.01,
            "dropout_rate": 0.1
        }
    },
    "distilroberta": {
        "model_url": "distilroberta-base",
        "tokenizer_url": "distilroberta-base",
        "tokenizer_max_length": 384,
        "prepare_model_input_fn": prepare_input_albert,
        "span_model": ParametricBertModelQA,
        "train_params": {
            "epochs": 2,
            "initial_lr": 0.00003,
            "batch_size_train": 8,
            "batch_size_val": 8,
            "batch_size_test": 8,
            "weight_decay": 0.01,
            "dropout_rate": 0.1
        }
    },
    "distilroberta_extra_linear": {
        "model_url": "distilroberta-base",
        "tokenizer_url": "distilroberta-base",
        "tokenizer_max_length": 384,
        "prepare_model_input_fn": prepare_input_albert,
        "span_model": ExtraParametricBertModelQA,
        "train_params": {
            "epochs": 2,
            "initial_lr": 0.00003,
            "batch_size_train": 8,
            "batch_size_val": 8,
            "batch_size_test": 8,
            "weight_decay": 0.01,
            "dropout_rate": 0.1
        }
    },
    "bert": {
        "model_url": "bert-base-uncased",
        "tokenizer_url": "bert-base-uncased",
        "tokenizer_max_length": 384,
        "prepare_model_input_fn": prepare_input_albert,
        "span_model": ParametricBertModelQA,
        "train_params": {
            "epochs": 2,
            "initial_lr": 0.00003,
            "batch_size_train": 8,
            "batch_size_val": 8,
            "batch_size_test": 8,
            "weight_decay": 0.01,
            "dropout_rate": 0.1
        }
    }
}
