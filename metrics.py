from math import sqrt
from scipy.stats import pearsonr
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, precision_recall_curve
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

def mae(y, f):
    return mean_absolute_error(y, f)


def rmse(y, f):
    rmse = sqrt(mean_squared_error(y, f))
    return rmse


def mse(y, f):
    mse = mean_squared_error(y, f)
    return mse


def pearson(y, f):
    rp = pearsonr(y.flatten(), f.flatten())[0]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def r2(y, f):
    return r2_score(y, f)

def get_metrics(y,f):
    return mse(y, f), rmse(y, f), mae(y, f), r2(y, f), pearson(y, f), spearman(y, f)

def accuracy(y_pred, y_true):
    return metrics.accuracy_score(y_pred=y_pred.round(), y_true=y_true)

def precision(y_pred, y_true):
    return metrics.precision_score(y_pred=y_pred.round(), y_true=y_true)

def recall(y_pred, y_true):
    return metrics.recall_score(y_pred=y_pred.round(), y_true=y_true)

def roc_auc(y_pred, y_true):
    return metrics.roc_auc_score(y_score=y_pred, y_true=y_true)

def pr_auc(y_pred, y_true):
    return metrics.average_precision_score(y_score=y_pred, y_true=y_true)

def get_metrics_cf(y,f):
    return  roc_auc(y,f), pr_auc(y, f), accuracy(y,f), precision(y,f), recall(y,f)








def f1_score(y_pred, y_true):
    return metrics.f1_score(y_pred=y_pred.round(), y_true=y_true)