import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_accuracy(predictions, labels):
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    return accuracy_score(labels, predictions)


def compute_metrics(predictions, labels, average='macro'):
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average=average, zero_division=0),
        'precision': precision_score(labels, predictions, average=average, zero_division=0),
        'recall': recall_score(labels, predictions, average=average, zero_division=0)
    }

    return metrics
