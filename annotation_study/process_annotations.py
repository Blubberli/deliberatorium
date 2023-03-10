import pandas as pd
import itertools
import json
import operator
from collections import Counter
from utils import read_all_maps
from util import remove_url_and_hashtags
from evaluate_study import convert_stringlabels_to_int
import numpy as np

email2USER = {"jan.angermeier@gmail.com": "Jan", "st169587@ims.uni-stuttgart.de": "victoria",
              "st169587@stud.uni-stuttgart.de": "victoria", "seemabhassan97@gmail.com": "seemab",
              "seemabhassan97@gmail.comco":"seemab"}


def convert_google_answers(df, info, counter, targets, seemab1=True, vic1=False):
    # return the dataframe with each candidate, target, candidate ID and target ID, annotation from each annotator and the confidence rating for each instance
    # return the current instance (=counter)
    lasti = 2
    jan = df[df["Username"] == "jan.angermeier@gmail.com"]
    if vic1:
        victoria = df[df["Username"] == "st169587@ims.uni-stuttgart.de"]
    else:
        victoria = df[df["Username"] == "st169587@stud.uni-stuttgart.de"]
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
        targetID = original_instance['corrected_child_ID'].values[0]
        candidateIDS = original_instance["corrected_cand_IDs"].values[0].split("##")
        # print to check that annotations from answer sheet match the text of the original instance
        # print(annotations)
        # print(original_instance)
        annotations = df.columns[lasti:i].values
        for j in range(len(annotations)):
            # holds child with candidate
            col = annotations[j]
            print(col)
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


def merge_annotations_final_100():
    targets = pd.read_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_targets.csv",
                          sep="\t")
    df1 = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo Annotations 5.csv",
        sep=",")
    df2 = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo Annotations 6.csv",
        sep=",")

    info = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo200instances_corrected.tsv",
        sep="\t")
    info = info[info['form questions'] != "Confidence Rating"]

    answers = [df1, df2]
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
    final_df.to_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/annotations5-6.csv",
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
    annotations = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo200instances_corrected.tsv",
        sep="\t",
        dtype=str)
    data_path = "/Users/falkne/PycharmProjects/deliberatorium/data/kialoV2/english"
    print(annotations.columns)
    argument_maps = read_all_maps(data_path)
    affected_ids = set([int(el.split(".")[0]) for el in annotations.corrected_child_ID.values])
    print(affected_ids)
    affected_maps = [map for map in argument_maps if map.id in affected_ids]
    id2map = {}
    for map in affected_maps:
        id2map[map.id] = map
    final_dic = {}
    print(annotations.columns)
    for i in range(len(annotations)):
        child_id = annotations.corrected_child_ID.values[i]
        candidate_id = annotations.corrected_cand_IDs.values[i]
        map = id2map[int(child_id.split(".")[0])]
        candidate_node_ids = candidate_id.split("##")
        candidate_nodes = [el for el in map.all_children if
                           el.id in candidate_node_ids]

        child_node = [el for el in map.all_children if
                      str(el.id) == str(child_id)][0]
        child_text = remove_url_and_hashtags(child_node.name)
        target_type = child_node.type
        parent_id = child_node.parent.id
        final_dic[child_id] = {
            "candidates": {}, "text": child_text, "parent ID": str(parent_id), "nodeType": target_type,
            "map_claim": map.name}
        for i in range(len(candidate_nodes)):
            cand = candidate_nodes[i]
            final_dic[child_id]["candidates"][cand.id] = {"text": remove_url_and_hashtags(cand.name)}
    with open('/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/child_and_candidate_info.json',
              'w', encoding='utf8') as json_file:
        json.dump(final_dic, json_file, indent=2)


def create_candidate_with_target_file_with_backup():
    annotations = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo200instances_corrected.tsv",
        sep="\t",
        dtype=str)
    data_path = "/Users/falkne/PycharmProjects/deliberatorium/data/kialoV2/english"
    for i in range(len(annotations)):
        child_id = annotations.corrected_child_ID.values[i]
        candidate_id = annotations.corrected_cand_IDs.values[i]
        candidate_ids = candidate_id.split("##")
        if child_id in set(candidate_ids):
            print(child_id)


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
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/annotations5-6.csv",
        sep="\t", dtype=str)

    # print(annotations.candidateID.values)
    for instance, info in input_json.items():
        parentID = info["parent ID"]
        candidates = info["candidates"]
        candidate_ids = list(candidates.keys())

        predictions = []
        rank = False
        candidate2score = {}
        for candID, candInfo in candidates.items():
            labels_candidate = annotations[annotations["candidateID"] == candID]
            # print(len(labels_candidate))
            annotation_jan = labels_candidate.jan.values[0]
            annotation_vic = labels_candidate.victoria.values[0]
            annotation_seemab = labels_candidate.seemab.values[0]
            majority = Counter([annotation_jan, annotation_vic, annotation_seemab]).most_common(1)
            majority = majority[0][0]
            score = np.mean(convert_stringlabels_to_int([annotation_jan, annotation_vic, annotation_seemab]))
            cand = {"text": candInfo["text"], "id": candID, "labelJan": annotation_jan, "labelVictoria": annotation_vic,
                    "labelSeemab": annotation_seemab, "labelMajority": majority, "avg_score": score}
            predictions.append(cand)
            candidate2score[candID] = score
            # compute rank
            if candID == parentID:
                rank = True
                rank_jan = get_rank_annotator(annotation_jan)
                rank_vic = get_rank_annotator(annotation_vic)
                rank_seemab = get_rank_annotator(annotation_seemab)
                rank_majority = get_rank_annotator(majority)
        sorted_scores = sorted(candidate2score.items(), key=operator.itemgetter(1))
        scored_rank = None
        for i in range(len(sorted_scores)):
            if sorted_scores[i][0] == parentID:
                scored_rank = i + 1
        if rank:
            info["predictions"] = predictions
            info["rank_victoria"] = rank_vic
            info["rank_jan"] = rank_jan
            info["rank_seemab"] = rank_seemab
            info["rank_majority"] = rank_majority
            info["rank_score_based"] = scored_rank
    json.dump(input_json,
              open(
                  "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/annotation_results_second_part.json",
                  "w", encoding='utf8'))
    # print(instance, parentID)


def add_corrected_ID():
    df = pd.read_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo200instances.tsv",
                     sep="\t")
    print(len(df))
    targets = pd.read_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_targets.csv",
                          sep="\t")
    counter = 0
    print(len(targets))
    real_ids = list(targets.ID.values)
    newIDs = []
    print(df.columns)
    for id in real_ids:
        newIDs.append([id, id])
    newIDs = list(itertools.chain(*newIDs))

    df["claimID"] = newIDs
    df.to_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/Kialo200instances2.tsv",
              sep="\t", index=False)


def get_child2annotations_dic(annotation_results):
    from collections import defaultdict
    child2annot_dic = defaultdict(list)
    child2info = defaultdict(dict)
    for index, row in annotation_results.iterrows():
        child_id = row.childID
        candidateID = row.candidateID
        parentID = row.parentID
        child2info[child_id]["parentID"] = parentID
        child2info[child_id]["conf_jan"] = row.conf_jan
        child2info[child_id]["conf_seemab"] = row.conf_seemab
        child2info[child_id]["conf_vic"] = row.conf_vic
        child2info[child_id]["child_text"] = row.child_text
        child2annot_dic[child_id].append(
            {"candidateID": candidateID, "candidate_text": row.candidate_text, "annot_jan": row.annot_jan,
             "annot_seemab": row.annot_seemab, "annot_vic": row.annot_vic})
    return child2annot_dic, child2info


def create_annotation_json_final():
    annotation_results = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_responses/annotation_data_finalized_corrected.tsv",
        sep="\t", dtype=str)
    print(annotation_results.columns)
    child2annotations, child2info = get_child2annotations_dic(annotation_results)
    data_path = "/Users/falkne/PycharmProjects/deliberatorium/data/kialoV2/english"
    print(len(child2annotations.keys()))
    argument_maps = read_all_maps(data_path)
    id2map = {}
    for map in argument_maps:
        id2map[map.id] = map
    json_dic = {}
    for childid, childinfo in child2info.items():
        info = {}
        predictions = []
        candidate2score = {}

        map = id2map[int(childid.split(".")[0])]
        print(map)
        print(childid)
        claim = map.name
        child_node = [el for el in map.all_children if str(el.id) == str(childid)][0]
        level = child_node.get_level()
        target_type = child_node.type
        candidate_info = child2annotations[childid]

        info["text"] = childinfo["child_text"]
        info["parent ID"] = childinfo["parentID"]
        info["nodeType"] = target_type
        info["depth"] = level
        info["map_claim"] = claim

        info["conf_jan"] = childinfo["conf_jan"]
        info["conf_vic"] = childinfo["conf_vic"]
        info["conf_seemab"] = childinfo["conf_seemab"]

        for candidate in candidate_info:

            majority = Counter([candidate["annot_jan"], candidate["annot_seemab"], candidate["annot_vic"]]).most_common(
                1)
            majority = majority[0][0]
            score = np.mean(convert_stringlabels_to_int(
                [candidate["annot_jan"], candidate["annot_seemab"], candidate["annot_vic"]]))

            candidate["avg_score"] = score
            candidate["labelMajority"] = majority
            candidate2score[candidate["candidateID"]] = score

            if candidate["candidateID"] == str(childinfo["parentID"]):
                rank_jan = get_rank_annotator(candidate["annot_jan"])
                rank_vic = get_rank_annotator(candidate["annot_seemab"])
                rank_seemab = get_rank_annotator(candidate["annot_vic"])
                rank_majority = get_rank_annotator(majority)
            predictions.append(candidate)

        sorted_scores = sorted(candidate2score.items(), key=operator.itemgetter(1))
        for i in range(len(sorted_scores)):
            if sorted_scores[i][0] == childinfo["parentID"]:
                scored_rank = i + 1
        info["predictions"] = predictions
        info["rank_victoria"] = rank_vic
        info["rank_jan"] = rank_jan
        info["rank_seemab"] = rank_seemab
        info["rank_majority"] = rank_majority
        info["rank_score_based"] = scored_rank
        json_dic[childid] = info
    with open(
            '/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_responses/annotation_results_final.json',
            'w', encoding='utf8') as json_file:
        json.dump(json_dic, json_file, indent=2)

        # compute ranks for performance

    # child ID
    # dic: "candidates : {"id: {"text": "textcontent"}
    # "text": "text content"
    # "parent ID": parentIDval
    # "nodeType": nodetypeVal
    # rank for each annot
    # rank based on score
    # predictions: [
    # {text: text, "id", labelJan, labelVictoria, labelSeemab, labelMajority, avgscore}


def fix_annotations():
    annotations_orig = pd.read_csv(
        "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_responses/annotation_data_finalized.tsv",
        sep="\t")

    print(annotations_orig.columns)
    child_texts = annotations_orig.child_text.values
    print(len(child_texts))
    print(len(set(annotations_orig.childID.values)))
    print(Counter(annotations_orig.childID.values))
    print(Counter(child_texts))
    from collections import defaultdict
    candidate2annotations = {}
    for i in range(1, 8):
        df = pd.read_csv(
            "/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_responses/Kialo Annotations %d.tsv" % i,
            sep="\t")

        df = df[df.Username != "Confidence Rating"]
        df = df[df.Username != "What led you to choose this as the best parent?"]
        annotators = df.columns[1:]
        used_children = set()
        firstbug = False
        for index, row in df.iterrows():
            inputtext = row.Username
            for annot in annotators:
                annotation = row[annot]
                annotator = email2USER[annot]
                mychild = None
                for child in child_texts:
                    if child in inputtext:
                        candidate = inputtext.replace(child, "")[2:-1]
                        mychild = child
                        if "It is couples and families, not women alone, who make the decision. It just so happens that nature dictates it is a woman who must carry the child" in mychild:
                            if "Even in the west, women still don't have equal access to many of their rights, particularly [when it comes to healthcare]" in candidate and not firstbug:
                                thischild = "12304.1281"
                                mychild = mychild+"#"+thischild
                                firstbug = True
                        if mychild not in candidate2annotations:
                            candidate2annotations[mychild] = defaultdict(dict)
                        candidate2annotations[mychild][candidate][annotator] = annotation

    new_annot_jan = []
    new_annot_vic = []
    new_annot_seemab = []
    for i in range(len(annotations_orig)):
        childID = annotations_orig.childID[i]
        if childID == "12304.1281":
            childtext = annotations_orig.child_text.values[i]+"#"+"12304.1281"
        else:
            childtext = annotations_orig.child_text.values[i]
        candidatetext = annotations_orig.candidate_text.values[i]
        allchildannots = candidate2annotations[childtext]
        candidate_annots = allchildannots[candidatetext]
        if not candidate_annots:
            new_annot_jan.append(None)
            new_annot_vic.append(None)
            new_annot_seemab.append(None)
        else:
            new_annot_jan.append(candidate_annots["Jan"])
            new_annot_vic.append(candidate_annots["victoria"])
            new_annot_seemab.append(candidate_annots["seemab"])
    print(len(new_annot_vic))
    print(len(new_annot_seemab))
    print(len(new_annot_jan))
    annotations_orig["annot_jan"] = new_annot_jan
    annotations_orig["annot_seemab"] = new_annot_seemab
    annotations_orig["annot_vic"] = new_annot_vic
    annotations_orig.to_csv("/Users/falkne/PycharmProjects/deliberatorium/data/annotation200instances/all_responses/annotation_data_finalized_corrected.tsv", sep
                            ="\t", index=False)

#A library of decisions would allow the Ethereum community to defend decisions much more effectively against brigading by allowing them to point to the Kialo debate or individual arguments therein, instead of having to formulate arguments themselves, which most people are unable to or not interested to be spending their precious time on.
#A library of decisions would allow the Ethereum community to defend decisions much more effectively against brigading by allowing them to point to the Kialo debate or individual arguments therein, instead of having to formulate arguments themselves, which most people are unable to or not interested to be spending their precious time on.



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
    # add_corrected_ID()
    # candidate_with_target_file()
    # create_candidate_with_target_file_with_backup()
    IDS_to_replace = ["2223.106"]
    create_annotation_json_final()
    #fix_annotations()
