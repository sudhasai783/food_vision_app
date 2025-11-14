# src/predictor.py
import torch
import numpy as np
import torch.nn.functional as F

def predict_topk(model, input_tensor, k=5):
    """
    Returns top-k predictions as a list of (class_index, probability).
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    topk_idx = np.argsort(probs)[-k:][::-1]
    return [(int(i), float(probs[i])) for i in topk_idx]


def predict_single(model, input_tensor):
    """
    Returns only the top prediction (index, probability).
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        top_idx = torch.argmax(probs, dim=1).item()
        top_prob = probs[0, top_idx].item()

    return int(top_idx), float(top_prob)
