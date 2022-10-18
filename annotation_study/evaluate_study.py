import itertools
import json

import pandas as pd
from skll.metrics import kappa, correlation
from sklearn.metrics import classification_report
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation import Evaluation as eval
from eval_util import format_metrics
from sklearn.preprocessing import StandardScaler

# agreement:
# average F1 Macro, average F1 for each class, spearman correlation, weighted kappa
# annotator2annotator matrix
# table with the average
ANNOTATORS = {"jan": "annot_jan", "victoria": "annot_vic", "seemab": "annot_seemab"}
ANNOT_CONF = {"jan": "conf_jan", "victoria": "conf_vic", "seemab": "conf_seemab"}


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


def convert_stringlabels_to_binary(labels):
    """return the list of string labels as a list of only two labels"""
    new_labels = []
    for label in labels:
        if label != 3:
            new_labels.append(0)
        else:
            new_labels.append(1)
    return new_labels


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


def get_agreement_metrics(results, annotator1, annotator2):
    """get a dictionary with all agreement metrics for two annotators"""
    annotator_label1 = ANNOTATORS[annotator1]
    annotator_label2 = ANNOTATORS[annotator2]
    result_dic = {}
    # to check if there are missing annotations
    # print("number of annotations used : %d" % len(results))

    predictions = [vals["predictions"] for child, vals in results.items()]
    # print(predictions)
    annotations_1 = [[pred[annotator_label1] for pred in el] for el in predictions]
    annotations_2 = [[pred[annotator_label2] for pred in el] for el in predictions]
    annotations_1 = list(itertools.chain(*annotations_1))
    annotations_2 = list(itertools.chain(*annotations_2))

    confidence_scores_annot1 = [vals[ANNOT_CONF[annotator1]] for child, vals in results.items()]

    confidence_scores_annot2 = [vals[ANNOT_CONF[annotator2]] for child, vals in results.items()]
    confidence_scores_annot1 = [int(el) for el in confidence_scores_annot1]
    confidence_scores_annot2 = [int(el) for el in confidence_scores_annot2]

    annotations_as_int_1 = convert_stringlabels_to_int(annotations_1)
    annotations_as_int_2 = convert_stringlabels_to_int(annotations_2)

    report = classification_report(y_true=annotations_1, y_pred=annotations_2, output_dict=True)
    result_dic["LESS SUITABLE PARENT"] = report["LESS SUITABLE PARENT"]["f1-score"]
    result_dic["BEST PARENT"] = report["BEST PARENT"]["f1-score"]
    result_dic["SUITABLE PARENT"] = report["SUITABLE PARENT"]["f1-score"]
    result_dic["F1Macro"] = report["macro avg"]["f1-score"]
    result_dic["CONFIDENCE CORR"] = correlation(y_true=confidence_scores_annot1,
                                                y_pred=confidence_scores_annot2, corr_type="spearman")
    weighted_kappa = get_weighted_kappa(weights=[1, 2, 5], annot1=annotations_as_int_1,
                                        annot2=annotations_as_int_2)
    binary_kappa = kappa(y_true=convert_stringlabels_to_binary(annotations_as_int_1),
                         y_pred=convert_stringlabels_to_binary(annotations_as_int_2))
    corr = correlation(y_true=annotations_as_int_1, y_pred=annotations_as_int_2, corr_type="spearman")
    result_dic["wKappa"] = weighted_kappa
    result_dic["KappaBINARY"] = binary_kappa
    result_dic["spearman"] = corr
    return result_dic


def get_average_frame(json_file):
    """create a dataframe with the average of all metrics (average for each metric), metrics are always computed in a pair-wise fashion"""
    metrics_dic = defaultdict(list)
    annotation_results = json.load(open(json_file))
    annotators = list(ANNOTATORS.keys())
    for annot_index1 in range(len(annotators)):
        for annot_index2 in range(len(annotators)):
            if annot_index1 != annot_index2:
                annot1 = annotators[annot_index1]
                annot2 = annotators[annot_index2]
                agreement = get_agreement_metrics(annotation_results, annot1, annot2)
                for k, v in agreement.items():
                    metrics_dic[k].append(v)
    average_dic = {}
    for k, v in metrics_dic.items():
        average_dic[k] = np.average(np.array(v))
    avg_df = pd.DataFrame({'metric': average_dic.keys(), 'average': average_dic.values()})
    return avg_df


def create_agreement_matrices(json_file):
    """for each metric, create a 3x3 matrix with the pair-wise agreemet scores"""
    annotation_results = json.load(open(json_file))
    agreement_matrices = {"wKappa": np.zeros(shape=[3, 3]), "KappaBINARY": np.zeros(shape=[3, 3]),
                          "spearman": np.zeros(shape=[3, 3]),
                          "F1Macro": np.zeros(shape=[3, 3]), "SUITABLE PARENT": np.zeros(shape=[3, 3]),
                          "LESS SUITABLE PARENT": np.zeros(shape=[3, 3]), "BEST PARENT": np.zeros(shape=[3, 3]),
                          "CONFIDENCE CORR": np.zeros(shape=[3, 3])}
    annotators = list(ANNOTATORS.keys())
    for annot_index1 in range(len(annotators)):
        for annot_index2 in range(len(annotators)):
            if annot_index1 != annot_index2:
                annot1 = annotators[annot_index1]
                annot2 = annotators[annot_index2]
                agreement = get_agreement_metrics(annotation_results, annot1, annot2)
                for k, v in agreement.items():
                    agreement_matrices[k][annot_index1][annot_index2] = v
    return agreement_matrices


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


def compute_human_performance(json_file, nodeType=None):
    predictions = json.load(open(json_file))
    ranks_jan = []
    ranks_victoria = []
    ranks_seemab = []
    ranks_majority = []
    ranks_score = []
    for instance, info in predictions.items():
        node_type = info["nodeType"]
        if nodeType:
            if node_type == nodeType:
                ranks_jan.append(info["rank_jan"])
                ranks_victoria.append(info["rank_victoria"])
                ranks_seemab.append(info["rank_seemab"])
                ranks_majority.append(info["rank_majority"])
                ranks_score.append(info["rank_score_based"])
        else:
            ranks_jan.append(info["rank_jan"])
            ranks_victoria.append(info["rank_victoria"])
            ranks_seemab.append(info["rank_seemab"])
            ranks_majority.append(info["rank_majority"])
            ranks_score.append(info["rank_score_based"])

    print("Jan")
    print(format_metrics(eval.calculate_metrics(ranks_jan)))
    print("Victoria")
    print(format_metrics(eval.calculate_metrics(ranks_victoria)))
    print("Seemab")
    print(format_metrics(eval.calculate_metrics(ranks_seemab)))
    print("majority")
    print(format_metrics(eval.calculate_metrics(ranks_majority)))
    print("score based")
    print(format_metrics(eval.calculate_metrics(ranks_score)))


def get_standardized_confidence_scores(jsonfile):
    annotation_results = json.load(open(jsonfile))
    confidence_scores_annot1 = [vals[ANNOT_CONF["jan"]] for child, vals in annotation_results.items()]

    confidence_scores_annot2 = [vals[ANNOT_CONF["seemab"]] for child, vals in annotation_results.items()]
    confidence_scores_annot3 = [vals[ANNOT_CONF["victoria"]] for child, vals in annotation_results.items()]

    confidence_scores_annot1 = [int(el) for el in confidence_scores_annot1]
    confidence_scores_annot2 = [int(el) for el in confidence_scores_annot2]
    confidence_scores_annot3 = [int(el) for el in confidence_scores_annot3]
    all_scores = np.array([confidence_scores_annot1, confidence_scores_annot2, confidence_scores_annot3])
    all_scores = StandardScaler().fit_transform(all_scores)
    all_corrs = []
    for i in range(all_scores.shape[0]):
        row_1 = all_scores[i, :]
        print(len(row_1))
        for j in range(all_scores.shape[0]):
            if i != j:
                row_2 = all_scores[j, :]
                corr = correlation(y_true=row_1,
                                   y_pred=row_2, corr_type="spearman")
                all_corrs.append(corr)
    print(all_corrs)
    print(np.average(all_corrs))


def plot_ranks(jsonfile):
    annotation_results = json.load(open(jsonfile))
    ranks_score = []
    ranks_jan = []
    ranks_victoria = []
    ranks_seemab = []
    lowest_ranks = []
    for instance, info in annotation_results.items():
        node_type = info["nodeType"]
        ranks_jan.append(info["rank_jan"])
        ranks_victoria.append(info["rank_victoria"])
        ranks_seemab.append(info["rank_seemab"])
        ranks_score.append(info["rank_score_based"])
        lowest_ranks.append(min([info["rank_jan"], info["rank_victoria"], info["rank_seemab"]]))

    print(ranks_score)
    from collections import Counter
    print(Counter(ranks_score))
    print(format_metrics(eval.calculate_metrics(lowest_ranks)))


def extract_examples_nononegot(jsonfile):
    annotation_results = json.load(open(jsonfile))
    annotation_tsv = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_responses/annotation_data_finalized_corrected.tsv",
        sep="\t", dtype=str)
    print(annotation_tsv.columns)
    wrong_instances = []
    for instance, info in annotation_results.items():
        if info["rank_jan"] > 5 and info["rank_seemab"] > 5 and info["rank_victoria"] > 5:
            wrong_instances.append(instance)
    df = annotation_tsv[annotation_tsv["childID"].isin(wrong_instances)]
    df.to_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_responses/results/hardest_examples.csv",
        sep="\t", index=False)


def confidence_vs_ranks(jsonfile):
    annotation_results = json.load(open(jsonfile))
    confidence_scores_jan = [int(vals[ANNOT_CONF["jan"]]) for child, vals in annotation_results.items()]
    confidence_scores_seemab = [int(vals[ANNOT_CONF["seemab"]]) for child, vals in annotation_results.items()]
    confidence_scores_victoria = [int(vals[ANNOT_CONF["victoria"]]) for child, vals in annotation_results.items()]

    ranks_jan = [int(vals["rank_jan"]) for child, vals in annotation_results.items()]
    ranks_seemab = [int(vals["rank_seemab"]) for child, vals in annotation_results.items()]
    ranks_victoria = [int(vals["rank_victoria"]) for child, vals in annotation_results.items()]
    print("Jan:")
    print(correlation(y_true=confidence_scores_jan,
                      y_pred=ranks_jan, corr_type="spearman"))
    print("seemab:")
    print(correlation(y_true=confidence_scores_seemab,
                      y_pred=ranks_seemab, corr_type="spearman"))
    print("victoria")
    print(correlation(y_true=confidence_scores_victoria,
                      y_pred=ranks_victoria, corr_type="spearman"))


def plot_confidence_scores(jsonfile):
    annotation_results = json.load(open(jsonfile))
    confidence_scores_jan = [int(vals[ANNOT_CONF["jan"]]) for child, vals in annotation_results.items()]
    confidence_scores_seemab = [int(vals[ANNOT_CONF["seemab"]]) for child, vals in annotation_results.items()]
    confidence_scores_victoria = [int(vals[ANNOT_CONF["victoria"]]) for child, vals in annotation_results.items()]
    confidence_average = [
        np.mean([confidence_scores_jan[i], confidence_scores_victoria[i], confidence_scores_seemab[i]])
        for i in range(len(confidence_scores_victoria))]
    # kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})
    # fig, ax = plt.subplots(1, 2)
    sns.countplot(confidence_average, color="dodgerblue", label="average")
    # sns.countplot(confidence_scores_seemab, color="orange", label="A2", ax=ax[1])
    # sns.countplot(confidence_scores_victoria, color="deeppink", label="A3", ax=ax[2])
    plt.legend()
    plt.show()


def agreement_confidencebins(jsonfile, annot1, annot2):
    annotation_results = json.load(open(jsonfile))
    annotator_label1 = ANNOTATORS[annot1]
    annotator_label2 = ANNOTATORS[annot2]
    confidence_scores_annot1 = [vals[ANNOT_CONF[annot1]] for child, vals in annotation_results.items()]
    confidence_scores_annot2 = [vals[ANNOT_CONF[annot2]] for child, vals in annotation_results.items()]
    average_confidence = [np.mean([int(confidence_scores_annot1[i]), int(confidence_scores_annot2[i])]) for i in
                          range(len(confidence_scores_annot2))]

    predictions = [vals["predictions"] for child, vals in annotation_results.items()]
    # print(predictions)
    annotations_1 = [[pred[annotator_label1] for pred in el] for el in predictions]
    annotations_2 = [[pred[annotator_label2] for pred in el] for el in predictions]
    thresh = np.median(average_confidence)
    annotations_1_lowconf = [annotations_1[i] for i in range(len(average_confidence)) if average_confidence[i] <= thresh]
    annotations_2_lowconf = [annotations_2[i] for i in range(len(average_confidence)) if average_confidence[i] <= thresh]
    print("low conf items %d " % len(annotations_2_lowconf))
    annotations_1_lowconf = list(itertools.chain(*annotations_1_lowconf))
    annotations_2_lowconf = list(itertools.chain(*annotations_2_lowconf))
    annotations_1_lowconf = convert_stringlabels_to_int(annotations_1_lowconf)
    annotations_2_lowconf = convert_stringlabels_to_int(annotations_2_lowconf)
    weighted_kappa = get_weighted_kappa(weights=[1, 2, 5], annot1=annotations_1_lowconf,
                                        annot2=annotations_2_lowconf)
    print("weighted kappa for low confidence is %.3f" % weighted_kappa)

    annotations_1_highconf = [annotations_1[i] for i in range(len(average_confidence)) if average_confidence[i] > thresh]
    annotations_2_highconf = [annotations_2[i] for i in range(len(average_confidence)) if average_confidence[i] > thresh]
    print("high conf items %d " % len(annotations_1_highconf))
    annotations_1_highconf = list(itertools.chain(*annotations_1_highconf))
    annotations_2_highconf = list(itertools.chain(*annotations_2_highconf))
    annotations_1_highconf = convert_stringlabels_to_int(annotations_1_highconf)
    annotations_2_highconf = convert_stringlabels_to_int(annotations_2_highconf)
    weighted_kappa = get_weighted_kappa(weights=[1, 2, 5], annot1=annotations_1_highconf,
                                        annot2=annotations_2_highconf)
    print("weighted kappa for high confidence is %.3f" % weighted_kappa)


if __name__ == '__main__':
    jsonfile = "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_responses/annotation_results_final.json"
    m = create_agreement_matrices(jsonfile)
    df = get_average_frame(jsonfile)
    df.to_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_responses/results/avg_agreement.csv",
        sep="\t", index=False)

    for annot1 in ANNOTATORS:
        for annot2 in ANNOTATORS:
            if annot1 != annot2:
                print("agreement between %s and %s" % (annot1, annot2))
                agreement_confidencebins(jsonfile, annot1, annot2)
    # compute_human_performance(jsonfile)
    # print("\n")
    # compute_human_performance(jsonfile, 1)
    # print("\n")
    # compute_human_performance(jsonfile, -1)

    # for metric, matrix in m.items():
    #    plot_agreement_matrices(matrix, title=metric,
    #                            out_path="/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_responses/results/plots/%s.png" % metric)
