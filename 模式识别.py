import math
def calculate_metrics(y_true, y_pred, y_score=None):
    TP = FP = FN = TN = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            TP += 1
        elif t == 0 and p == 1:
            FP += 1
        elif t == 1 and p == 0:
            FN += 1
        elif t == 0 and p == 0:
            TN += 1
    P = TP / (TP + FP) if (TP + FP) > 0 else 0
    R = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    AUC = calculate_auc(y_true, y_score) if y_score else None
    return P, R, F1, AUC
def calculate_auc(y_true, y_score):
    data = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    pos_count = sum(y_true)
    neg_count = len(y_true) - pos_count
    if pos_count == 0 or neg_count == 0:
        return 0.5
    area = 0
    prev_fpr = prev_tpr = 0
    TP = FP = 0
    for _, label in data:
        if label == 1:
            TP += 1
        else:
            FP += 1
        tpr = TP / pos_count
        fpr = FP / neg_count
        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr, prev_tpr = fpr, tpr
    return area
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 0, 1, 0, 1]
y_score = [0.9, 0.1, 0.4, 0.8, 0.2, 0.7]
P, R, F1, AUC = calculate_metrics(y_true, y_pred, y_score)
print(f"Precision: {P:.3f}")
print(f"Recall: {R:.3f}")
print(f"F1-score: {F1:.3f}")
print(f"AUC: {AUC:.3f}")