"""v_entropy and v_info from the authors' v_info.py (kawine/dataset_difficulty at 5852e81), for text-to-text models."""
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForSeq2SeqLM
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


def seq2seq_classifier(model, tokenizer, labels=("A", "B")):
    """DDB: stands in for pipeline('text-classification'): P(label) from the first decoder step, softmax over `labels`."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(model).cuda().eval()
    ids = [tokenizer(label).input_ids[0] for label in labels]

    @torch.no_grad()
    def classify(texts):
        enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        start = torch.full((len(texts), 1), model.config.decoder_start_token_id, device="cuda")
        probs = model(**enc, decoder_input_ids=start).logits[:, 0, ids].float().softmax(-1).tolist()
        return [[{"label": label, "score": p} for label, p in zip(labels, row)] for row in probs]

    return classify


def v_entropy(data_fn, model, tokenizer, input_key='sentence1', batch_size=100):
    """
    Calculate the V-entropy (in bits) on the data given in data_fn. This can be
    used to calculate both the V-entropy and conditional V-entropy (for the
    former, the input column would only have null data and the model would be
    trained on this null data).

    Args:
        data_fn: path to data; should contain the label in the 'label' column
        model: path to saved model or model name in HuggingFace library
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        input_key: column name of X variable in data_fn
        batch_size: data batch_size

    Returns:
        Tuple of (V-entropies, correctness of predictions, predicted labels).
        Each is a List of n entries (n = number of examples in data_fn).
    """
    # added for gpt2 
    if tokenizer == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model, pad_token_id=tokenizer.eos_token_id)

    classifier = seq2seq_classifier(model, tokenizer)  # DDB: FLAN-T5 answers "A"/"B" as text; it has no classifier head
    data = pd.read_csv(data_fn)
    
    entropies = []
    correct = []
    predicted_labels = []

    for j in tqdm(range(0, len(data), batch_size)):
        batch = data[j:j+batch_size]
        predictions = classifier(batch[input_key].tolist())

        for i in range(len(batch)):
            prob = next(d for d in predictions[i] if d['label'] == batch.iloc[i]['label'])['score']
            entropies.append(-1 * np.log2(prob))
            predicted_label = max(predictions[i], key=lambda x: x['score'])['label'] 
            predicted_labels.append(predicted_label)
            correct.append(predicted_label == batch.iloc[i]['label'])

    torch.cuda.empty_cache()

    return entropies, correct, predicted_labels


def v_info(data_fn, model, null_data_fn, null_model, tokenizer, out_fn="", input_key='sentence1'):
    """
    Calculate the V-entropy, conditional V-entropy, and V-information on the
    data in data_fn. Add these columns to the data in data_fn and return as a 
    pandas DataFrame. This means that each row will contain the (pointwise
    V-entropy, pointwise conditional V-entropy, and pointwise V-info (PVI)). By
    taking the average over all the rows, you can get the V-entropy, conditional
    V-entropy, and V-info respectively.

    Args:
        data_fn: path to data; should contain the label in the 'label' column 
            and X in column specified by input_key
        model: path to saved model or model name in HuggingFace library
        null_data: path to null data (column specified by input_key should have
            null data)
        null_model: path to saved model trained on null data
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        out_fn: where to saved 
        input_key: column name of X variable in data_fn 

    Returns:
        Pandas DataFrame of the data in data_fn, with the three additional 
        columns specified above.
    """
    data = pd.read_csv(data_fn)
    data['H_yb'], _, _ = v_entropy(null_data_fn, null_model, tokenizer, input_key=input_key) 
    data['H_yx'], data['correct_yx'], data['predicted_label'] = v_entropy(data_fn, model, tokenizer, input_key=input_key)
    data['PVI'] = data['H_yb'] - data['H_yx']

    if out_fn:
        data.to_csv(out_fn)

    return data

