import pandas as pd
import itertools
import json
from collections import Counter


def convert_google_answers(df, info, counter, targets, seemab1=True):
    # return the dataframe with each candidate, target, candidate ID and target ID, annotation from each annotator and the confidence rating for each instance
    # return the current instance (=counter)
    lasti = 2
    jan = df[df["Username"] == "jan.angermeier@gmail.com"]
    victoria = df[df["Username"] == "st169587@ims.uni-stuttgart.de"]
    if seemab1:
        seemab = df[df["Username"] == "seemabhassan97@gmail.com"]
    else:
        seemab = df[df["Username"] == "seemabhassan97@gmail.comco"]
    df_dic = {"target": [], "targetID": [], "candidateID": [], "jan": [], "victoria": [], "seemab": [], "jan_conv": [],
              "vic_conf": [], "seemab_conf": []}
    # iterate over dataframe columns, select a batch of 11, first 10 are the candidates and the 11th the confidence rating.
    # the value of each annotator is specified in the corresponding row (that holds their email)
    for i in range(13, len(df.columns) + 11, 11):
        # keep track of the instance we are requesting, the original instance with IDs is in the 200instances_df
        original_instance = info.iloc[[counter]]
        targetID = targets.iloc[[counter]]['ID'].values[0]
        candidateIDS = original_instance["candidateIDs"].values[0].split("##")
        # print to check that annotations from answer sheet match the text of the original instance
        # print(annotations)
        # print(original_instance)
        annotations = df.columns[lasti:i].values
        for j in range(len(annotations)):
            # holds child with candidate
            col = annotations[j]
            annot_jan = jan[col].values[0]
            annot_vic = victoria[col].values[0]
            annot_seemab = seemab[col].values[0]
            df_dic["targetID"].append(targetID)
            df_dic["target"].append(col)
            if j != 10:
                candidateID = candidateIDS[j]
                df_dic["candidateID"].append(candidateID)
                df_dic["jan"].append(annot_jan)
                df_dic["victoria"].append(annot_vic)
                df_dic["seemab"].append(annot_seemab)
                df_dic["jan_conv"].append("-")
                df_dic["vic_conf"].append("-")
                df_dic["seemab_conf"].append("-")
            else:
                # the 11th column holds the confidence rating
                df_dic["candidateID"].append("CONF")
                df_dic["jan"].append("CONF")
                df_dic["victoria"].append("CONF")
                df_dic["seemab"].append("CONF")
                df_dic["jan_conv"].append(annot_jan)
                df_dic["vic_conf"].append(annot_vic)
                df_dic["seemab_conf"].append(annot_seemab)

        counter += 1
        lasti = i
    return pd.DataFrame().from_dict(df_dic), counter


def merge_annotations1to4():
    targets = pd.read_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_targets.csv",
                          sep="\t")
    df1 = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/kialo_answers_200instances/Kialo Annotations 1.csv",
        sep=",")
    df2 = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/kialo_answers_200instances/Kialo Annotations 2.csv",
        sep=",")
    df3 = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/kialo_answers_200instances/Kialo Annotations 3.csv",
        sep=",")
    df4 = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/kialo_answers_200instances/Kialo Annotations 4.csv",
        sep=",")
    info = pd.read_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo200instances.tsv",
                       sep="\t")
    # info contains 200 instances wih claim, candidate (ids..)
    info = info[info['form questions'] != "Confidence Rating"]

    answers = [df1, df2, df3, df4]
    counter = 0
    newdfs = []
    for i in range(len(answers)):
        df = answers[i]
        if i != 1:
            newdf, counter = convert_google_answers(df, info, counter, targets)
        else:
            newdf, counter = convert_google_answers(df, info, counter, targets, False)
        newdfs.append(newdf)

    final_df = pd.concat(newdfs)
    # now for each instance addd annotator info (
    final_df.to_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/annotations1-4.csv",
                    sep="\t",
                    index=False)


def add_candidate_and_target_info(annotations, candidates, targets):
    candidate_distances = []
    parent_ids = []
    gold_labels = []
    topics = []
    coarse_level = []
    target_types = []
    jan_conf = []
    vic_conf = []
    seemab_conf = []
    # create a copy without confidence ratings
    annotations_copy = annotations.copy()
    annotations_copy = annotations_copy[annotations_copy["jan_conv"] == "-"]
    for i in range(len(annotations)):
        target_id = annotations.targetID.values[i]
        target_id = float(target_id)
        target_info = targets[targets["ID"] == target_id]
        parent_id = target_info.parentID.values[0]
        target_coarse_level = target_info.coarseLevel.values[0]
        target_type = target_info['target type'].values[0]
        topic = target_info.topic.values[0]
        candidate_id = annotations.candidateID.values[i]
        if candidate_id == "CONF":
            jan_conf.append(10 * [annotations["jan_conv"].values[i]])
            vic_conf.append(10 * [annotations.vic_conf.values[i]])
            seemab_conf.append(10 * [annotations.seemab_conf.values[i]])
        else:
            candidate_id = float(candidate_id)
            candidate_info = candidates[candidates["id"] == candidate_id]
            candidate_distance = candidate_info["distance"].values[0]
            candidate_distances.append(candidate_distance)
            parent_ids.append(parent_id)
            topics.append(topic)
            coarse_level.append(target_coarse_level)
            target_types.append(target_type)
            if parent_id == candidate_id:
                gold_labels.append("BEST PARENT")
            elif candidate_distance <= 3:
                gold_labels.append("SUITABLE PARENT")
            else:
                gold_labels.append("LESS SUITABLE PARENT")
    annotations_copy["gold"] = gold_labels
    annotations_copy["parentID"] = parent_ids
    annotations_copy["distance"] = candidate_distances
    annotations_copy["topic"] = topics
    annotations_copy["level"] = coarse_level
    annotations_copy["type"] = target_types
    annotations_copy["jan_conf"] = list(itertools.chain(*jan_conf))
    annotations_copy["vic_conf"] = list(itertools.chain(*vic_conf))
    annotations_copy["seemab_conf"] = list(itertools.chain(*seemab_conf))
    return annotations_copy


def candidate_with_target_file():
    candidates = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_candidates.csv", sep="\t")
    targets = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_targets.csv",
        sep="\t")
    annotations = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/annotations1-4.tsv", sep="\t")
    final_dic = {}
    for i in range(len(annotations)):
        target_id = annotations.targetID.values[i]
        target_id = float(target_id)
        target_info = targets[targets["ID"] == target_id]
        parent_id = target_info.parentID.values[0]
        target_type = target_info['target type'].values[0]
        candidate_id = annotations.candidateID.values[i]

        if candidate_id == "CONF":
            continue
        else:
            candidate_id = float(candidate_id)
            candidate_info = candidates[candidates["id"] == candidate_id]
            if target_id not in final_dic:
                final_dic[target_id] = {
                    "candidates": {candidate_id: {"text": candidate_info["cleaned_comments"].values[0]}
                                   }, "text": target_info["target node clean"].values[0],
                    "parent ID": str(parent_id),
                    "nodeType": target_type}
            else:
                final_dic[target_id]["candidates"][candidate_id] = {
                    "text": candidate_info["cleaned_comments"].values[0]}
    print(final_dic)
    json.dump(final_dic,
              open(
                  "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/target_and_candidate_info.json",
                  "w"))


def get_rank_annotator(label):
    if label == "BEST PARENT":
        return 1
    if label == "SUITABLE PARENT":
        return 2
    else:
        return 6


def create_annotation_json():
    input_json = json.load(
        open("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/target_and_candidate_info.json"))
    annotations = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/annotations1-4_withInfo.csv",
        sep="\t")
    annotations["candidateID"] = [str(el) for el in annotations.candidateID.values]
    print(annotations.columns)
    # print(annotations.candidateID.values)
    for instance, info in input_json.items():
        parentID = info["parent ID"]
        candidates = info["candidates"]
        predictions = []
        rank = False
        for candID, candInfo in candidates.items():
            labels_candidate = annotations[annotations["candidateID"] == candID]
            # print(len(labels_candidate))
            annotation_jan = labels_candidate.jan.values[0]
            annotation_vic = labels_candidate.victoria.values[0]
            annotation_seemab = labels_candidate.seemab.values[0]
            majority = Counter([annotation_jan, annotation_vic, annotation_seemab]).most_common(1)
            majority = majority[0][0]
            print(majority)
            cand = {"text": candInfo["text"], "id": candID, "labelJan": annotation_jan, "labelVictoria": annotation_vic,
                    "labelSeemab": annotation_seemab, "labelMajority": majority}
            predictions.append(cand)
            # compute rank
            if candID == parentID:
                rank = True
                rank_jan = get_rank_annotator(annotation_jan)
                rank_vic = get_rank_annotator(annotation_vic)
                rank_seemab = get_rank_annotator(annotation_seemab)
                rank_majority = get_rank_annotator(majority)
        if rank:
            info["predictions"] = predictions
            info["rank_victoria"] = rank_vic
            info["rank_jan"] = rank_jan
            info["rank_seemab"] = rank_seemab
            info["rank_majority"] = rank_majority
    json.dump(input_json,
              open("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/annotation_results.json",
                   "w"))
    # print(instance, parentID)


if __name__ == '__main__':
    # create_annotation_json()
    # annotations = pd.read_csv(
    #    "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/annotations1-4.tsv", sep="\t")
    # candidates = pd.read_csv(
    #    "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_candidates.csv", sep="\t")
    # targets = pd.read_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_targets.csv",
    #                      sep="\t")
    # df = add_candidate_and_target_info(annotations, candidates, targets)
    # df.to_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/annotations1-4_withInfo.csv",
    #          sep="\t", index=False)
    create_annotation_json()
