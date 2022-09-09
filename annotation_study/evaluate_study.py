import pandas as pd
from skll.metrics import kappa, correlation
from sklearn.metrics import classification_report
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation import Evaluation as eval
from collections import Counter

# agreement:
# average F1 Macro, average F1 for each class, spearman correlation, weighted kappa
# annotator2annotator matrix
# table with the average
ANNOTATORS = ["jan", "victoria", "seemab"]
ANNOT_CONF = ["jan_conf", "vic_conf", "seemab_conf"]


def get_weighted_kappa(weights, annot1, annot2):
    # return weighted kappa for two annotators, the labels are weighted differently
    weight_matrix = np.zeros(shape=[3, 3])

    weight_matrix[0][1] = weights[0]
    weight_matrix[0][2] = weights[2]
    weight_matrix[1][0] = weights[0]
    weight_matrix[1][2] = weights[1]
    weight_matrix[2][0] = weights[2]
    weight_matrix[2][1] = weights[1]
    return kappa(y_true=annot1, y_pred=annot2, weights=weight_matrix)


def convert_stringlabels_to_int(labels):
    """return the list of string labels as a list  of ints, such that they can be put into (non)linear relationships"""
    labeldic = {'BEST PARENT': 1, 'SUITABLE PARENT': 2, 'LESS SUITABLE PARENT': 3}
    return [labeldic[el] for el in labels]


def plot_agreement_matrices(matrix, title, out_path=None):
    # plot an agreement matrix as a heatmap for a metric
    sns.set(font_scale=1.5)
    sns.heatmap(matrix, xticklabels=ANNOTATORS, yticklabels=ANNOTATORS, annot=True, cmap="Blues").set_title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=800)
    else:
        plt.show()
    plt.clf()


def get_agreement_metrics(df, annot_index1, annot_index2):
    """get a dictionary with all agreement metrics for two annotators"""
    annotator1 = ANNOTATORS[annot_index1]
    annotator2 = ANNOTATORS[annot_index2]
    result_dic = {}
    # to check if there are missing annotations
    # print("number of annotations used : %d" % len(df))
    # print("NAN values to be replaced: annotator %s: %d; annotator %s: %d" % (
    #    annotator1, df[annotator1].isna().sum(), annotator2, df[annotator2].isna().sum()))
    # fill missing annotations with less suitable
    # df[annotator1] = df[annotator1].fillna("LESS SUITABLE PARENT")
    # df[annotator2] = df[annotator2].fillna("LESS SUITABLE PARENT")
    annotations_1 = list(df[annotator1].values)
    annotations_2 = list(df[annotator2].values)
    annotations_as_int_1 = convert_stringlabels_to_int(annotations_1)
    annotations_as_int_2 = convert_stringlabels_to_int(annotations_2)

    report = classification_report(y_true=annotations_1, y_pred=annotations_2, output_dict=True)
    result_dic["LESS SUITABLE PARENT"] = report["LESS SUITABLE PARENT"]["f1-score"]
    result_dic["BEST PARENT"] = report["BEST PARENT"]["f1-score"]
    result_dic["SUITABLE PARENT"] = report["SUITABLE PARENT"]["f1-score"]
    result_dic["F1Macro"] = report["macro avg"]["f1-score"]
    result_dic["CONFIDENCE CORR"] = correlation(y_true=list(df[ANNOT_CONF[annot_index1]].values),
                                                y_pred=list(df[ANNOT_CONF[annot_index2]].values), corr_type="spearman")
    weighted_kappa = get_weighted_kappa(weights=[1, 2, 5], annot1=annotations_as_int_1,
                                        annot2=annotations_as_int_2)
    corr = correlation(y_true=annotations_as_int_1, y_pred=annotations_as_int_2, corr_type="spearman")
    result_dic["wKappa"] = weighted_kappa
    result_dic["spearman"] = corr
    return result_dic


def get_average_frame(annotation_frame):
    """create a dataframe with the average of all metrics (average for each metric), metrics are always computed in a pair-wise fashion"""
    metrics_dic = defaultdict(list)
    for annot1 in range(len(ANNOTATORS)):
        for annot2 in range(len(ANNOTATORS)):
            if annot1 != annot2:
                agreement = get_agreement_metrics(annotation_frame, annot1, annot2)
                for k, v in agreement.items():
                    metrics_dic[k].append(v)
    average_dic = {}
    for k, v in metrics_dic.items():
        average_dic[k] = np.average(np.array(v))
    avg_df = pd.DataFrame({'metric': average_dic.keys(), 'average': average_dic.values()})
    return avg_df


def create_agreement_matrices():
    """for each metric, create a 3x3 matrix with the pair-wise agreemet scores"""
    agreement_matrices = {"wKappa": np.zeros(shape=[3, 3]), "spearman": np.zeros(shape=[3, 3]),
                          "F1Macro": np.zeros(shape=[3, 3]), "SUITABLE PARENT": np.zeros(shape=[3, 3]),
                          "LESS SUITABLE PARENT": np.zeros(shape=[3, 3]), "BEST PARENT": np.zeros(shape=[3, 3]),
                          "CONFIDENCE CORR": np.zeros(shape=[3, 3])}
    for annot1 in range(len(ANNOTATORS)):
        for annot2 in range(len(ANNOTATORS)):
            if annot1 != annot2:
                agreement = get_agreement_metrics(df, annot1, annot2)
                for k, v in agreement.items():
                    agreement_matrices[k][annot1][annot2] = v
    return agreement_matrices


def compute_human_performance(df):
    ranks = {"jan": [], "victoria": [], "seemab": []}
    results = defaultdict(dict)
    gold_labels = list(df.gold.values)
    for i in range(len(df)):
        gold_label = gold_labels[i]
        if gold_label != "BEST PARENT":
            continue
        for annot in ANNOTATORS:
            annotation = df[annot].values[i]
            if gold_label == "BEST PARENT" and annotation == "BEST PARENT":
                ranks[annot].append(1)
            if gold_label == "BEST PARENT" and annotation == "SUITABLE PARENT":
                ranks[annot].append(2)
            else:
                ranks[annot].append(1000)
    for annotator in ANNOTATORS:
        annotations = ranks[annotator]
        prec1 = eval.precision_at_rank(annotations, 1)
        prec5 = eval.precision_at_rank(annotations, 5)
        mrr = eval.mean_reciprocal_rank(annotations)
        results[annotator]["prec1"] = prec1
        results[annotator]["prec5"] = prec5
        results[annotator]["mrr"] = mrr

    return results


def filter_data_frame(df, column, value):
    df_copy = df.copy()
    return df_copy[df_copy[column] == value]


def get_average_performance(df):
    all_results = defaultdict(list)
    results = compute_human_performance(df)
    for k, v in results.items():
        for metric, val in v.items():
            all_results[metric].append(val)
    average_results = {}
    for k, v in all_results.items():
        average_results[k] = np.average(np.array(v))
    return pd.DataFrame({'metric': average_results.keys(), 'average': average_results.values()})
