import math

TP = 148334
TN = 91775
FP = 426
FN = 365


# dokladność
def calc_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + fp + fn + tn)


# czułość
def calc_sensitivity(tp, fn):
    return tp / (tp + fn)


#specyficzność
def calc_specificity(tn, fp):
    return tn / (fp + tn)


#precyzja
def calc_ppv(tp, fp):
    return tp / (tp + fp)


#F1 score
def calc_F1_score(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn)


def calc_balanced_accuracy(tp, tn, fp, fn):
    return (calc_sensitivity(tp, fn) + calc_specificity(tn, fp)) / 2


# swoistość, odsetek prawidziwie negatywnych
def calc_tnr(tn, fp):
    return tn / (fp + tn)


# negatywna wartość predykcyjna
def calc_npv(tn, fn):
    return tn / (fn + tn)


# współczynnik fałszywie ujemny
def calc_fnr(fn, tp):
    return fn / (tp + fn)


# Wskaźnik fałszywie dodatnich
def calc_fpr(fp, tn):
    return fp / (fp + tn)


# wskaźnik fałszywych odkryć
def calc_fdr(fp, tp):
    return fp / (fp + tp)


# współczynnik fałszywych pominięć
def calc_for(fn, tn):
    return fn / (fn + tn)


#MCC, Współczynnik korelacji Matthews
def calc_mcc(tn, tp, fp, fn):
    return (tn * tp - fp * fn) / math.sqrt((tn + fn) * (fp + tp) * (tn + fp) * (fn + tp))


print("accuracy            ", calc_accuracy(TP, TN, FP, FN))
print("sensitivity         ", calc_sensitivity(TP, FN))
print("specificity         ", calc_specificity(TN, FP))
print("ppv                 ", calc_ppv(TP, FP))
print("F1 - score          ", calc_F1_score(TP, FP, FN))
print("balanced_accuracy   ", calc_balanced_accuracy(TP, TN, FP, FN))

print("****************************************")

print("tnr                 ", calc_tnr(TN, FP))
print("npv                 ", calc_npv(TN, FN))
print("fnr                 ", calc_fnr(FN, TP))
print("fpr                 ", calc_fpr(FP, TN))
print("fdr                 ", calc_fdr(FP, TP))
print("for                 ", calc_for(FN, TN))
print("mcc                 ", calc_mcc(TN, TP, FP, FN))

