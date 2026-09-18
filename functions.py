import numpy as np

def cross_entropy(logits, targets): # built with AI assistance
    logit_maxes = logits.max(axis=1, keepdims=True)
    logits_norm = logits - logit_maxes
    counts = logits_norm.exp()
    counts_sum = counts.sum(axis=1, keepdims=True)
    log_probs = logits_norm - counts_sum.log()
    loss = -log_probs[np.arange(len(targets)), targets].mean()
    return loss

def softmax(logits, axis):
    logits_max = logits.max(axis=axis, keepdims=True)
    logits_norm = logits - logits_max
    logits_exp = logits_norm.exp()
    counts_sum = logits_exp.sum(axis=axis, keepdims=True)
    return logits_exp / counts_sum