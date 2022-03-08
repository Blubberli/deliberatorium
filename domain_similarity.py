import pandas as pd
import seaborn as sns
from argumentMap import KialoMap, DeliberatoriumMap
import os
import itertools
import operator
from pathlib import Path
import spacy
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# define the POS tags of content words (vs function words)
CONTENTWORD_POSTAGS = {"noun", "verb", "adj", "propn", "adv"}
IT_MAPS = {"Doparia sulla legge elettorale (m1)", "Doparia sulla legge elettorale (m2)", "Naples Biofuels Discussion"}
main_topics = {"religion", "gender", "enviroment", "animals", "democracy", "europe", "technology", "education",
               "history", "violence", "war", "business", "human rights", "health", "morality", "philosophy",
               "science"}


def read_vocab_from_map(map, only_content_words=False):
    """
    This method reads in all text from a map and creates a vocabulary object
    :param map: an ArgumentMap object
    :param only_content_words: if set to true the lemma list will only contain content words (adjectives, nouns..)
    :return: a list of lemmata extracted from the content of a map
    """
    all_text = [node.name for node in map.all_children]
    # use the medium model trained on web data from spacy, use english (translations have to be used for e.g. IT maps)
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    lemmas = []
    for doc in nlp.pipe(all_text, batch_size=20):
        lemma_list = [str(tok.lemma_).lower() for tok in doc]
        pos_tags = [str(tok.pos_).lower() for tok in doc]
        if only_content_words:
            lemma_list = [lemma_list[i] for i in range(len(pos_tags)) if pos_tags[i] in CONTENTWORD_POSTAGS]
        lemmas.append(lemma_list)
    lemmas = list(itertools.chain(*lemmas))
    return lemmas


def build_vectorizer(map2lemmas):
    """
    Create a vectorizer based on word frequency of a corpus. Use the content of all given maps and merge all docuemnts
    (aka maps) into one big corpus. Create the vectorizer based on that.
    :param map2lemmas: a dictionary containing the maps with a list of their lemmas to build the vectorizer on
    :return: a tf-idf vectorizer object, minimum frequency is 3
    """
    corpora = []
    for _, lemma_list in map2lemmas.items():
        as_string = " ".join(lemma_list)
        corpora.append(as_string)
    vectorizer = TfidfVectorizer(min_df=3)
    vectorizer.fit(corpora)
    return vectorizer


def get_map_lemmas(maps, translated_map_path=None):
    """Read all deliberatorium or kialo maps and return a dictionary
     containing each map-name and a list of corresponding lemmata"""
    maps2lemmas = {}
    for _, map in maps.items():
        if map.name in IT_MAPS and translated_map_path:
            lemmas = read_vocab_from_file(
                translated_map_path % map.name, True)
        else:
            lemmas = read_vocab_from_map(map, True)
        maps2lemmas[map.name] = lemmas
    return maps2lemmas


def read_vocab_from_file(path, only_content_words=False):
    """
    Read files and extract a list of lemmata
    :param path: the path to read the file from
    :param only_content_words: if only content words should be considered (adjectives, nouns...)
    :return: a list of lemmata
    """
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    lemmas = []
    all_text = open(path).readlines()
    all_text = [line.strip() for line in all_text]
    for doc in nlp.pipe(all_text, batch_size=20):
        lemma_list = [str(tok.lemma_).lower() for tok in doc]
        pos_tags = [str(tok.pos_).lower() for tok in doc]
        if only_content_words:
            lemma_list = [lemma_list[i] for i in range(len(pos_tags)) if pos_tags[i] in CONTENTWORD_POSTAGS]
        lemmas.append(lemma_list)
    lemmas = list(itertools.chain(*lemmas))
    return lemmas


def get_tfidf_vector(lemmas, vectorizer):
    """
    Extract the tf-idf vector of a document
    :param lemmas: a list of lemmata representing a document
    :param vectorizer: a vectorizer used to transform the document
    :return: the tf-idf vector of the given document
    """
    corpus = " ".join(lemmas)
    vector = vectorizer.transform([corpus]).toarray()
    return vector


def get_top_ranked_words(vector, n, vectorizer):
    """
    Extract words with the highest tf-idf score given a tf-idf vector of a document / map.
    :param vector: the tf-idf document vector of a corpus
    :param n: the number of most significant words to be returned
    :param vectorizer: the vectorizer object which contains the vocab index
    :return: the most significant (in terms of tf-idf) n words of the document /map
    """
    feature_array = np.array(vectorizer.get_feature_names())
    tfidf_sorting = np.argsort(vector.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:n]
    return top_n


def shared_vocab(lemmas_1, lemmas_2):
    """Return the set of shared vocabulary between two lists of lemmas"""
    shared_vocab = set(lemmas_1).intersection(set(lemmas_2))
    return shared_vocab


def tfidf_similairty(vector_1, vector_2):
    """Return the cosine similarity between two vectors"""
    similarity = cosine_similarity(vector_1, vector_2)
    return similarity


def read_all_maps():
    """read all maps from deliberatorium and kialo, return them as two dictionaries"""
    data_path = Path.home() / "data/e-delib/deliberatorium/maps/kialo_maps"
    maps = os.listdir(data_path)
    kialo_maps = {}
    for i in tqdm(range(len(maps))):
        map = maps[i]
        path = "%s/%s" % (data_path, map)
        argument_map = KialoMap(path)
        kialo_maps[argument_map.name.strip()] = argument_map

    data_path = Path.home() / "data/e-delib/deliberatorium/maps/deliberatorium_maps"
    maps = os.listdir(data_path)

    deliberatorium_maps = {}
    for map in maps:
        argument_map = DeliberatoriumMap("%s/%s" % (str(data_path), map))
        deliberatorium_maps[argument_map.name] = argument_map
    return kialo_maps, deliberatorium_maps


def tag_headmap_kialo(path_kialo2topics, n, outpath, cut_most_frequent=False):
    """Heatmap for the top 50 most frequent topics"""
    topics = pd.read_csv(path_kialo2topics, sep="\t")
    topics.dropna(inplace=True)
    topics["topic_tags"] = [[w.strip().lower() for w in el.split(",")] for el in topics.topics.values]
    top50 = dict(Counter(itertools.chain(*topics.topic_tags.values)).most_common(n))
    vocab = list(top50.keys())
    if cut_most_frequent:
        vocab = vocab[10:]
    print("vocab size is %d" % len(vocab))
    vocab_matrix = np.zeros(shape=[len(vocab), len(vocab)])
    vocab2index = dict(zip(vocab, range(len(vocab))))
    for map_tags in topics.topic_tags.values:
        for tag1 in map_tags:
            if tag1 in vocab2index:
                index1 = vocab2index[tag1]
            for tag2 in map_tags:
                if tag2 in vocab2index:
                    index2 = vocab2index[tag2]
                    if index1 != index2:
                        vocab_matrix[index1][index2] += 1

    sns.set(font_scale=0.5)
    ax = sns.heatmap(vocab_matrix, xticklabels=vocab, yticklabels=vocab)
    plt.tight_layout()
    plt.savefig(outpath, dpi=1500)


def create_domains(path_kialo2topics, outpath):
    topics = pd.read_csv(path_kialo2topics, sep="\t")
    topics.dropna(inplace=True)
    topics["topic_tags"] = [[w.strip().lower() for w in el.split(",")] for el in topics.topics.values]
    top50 = dict(Counter(itertools.chain(*topics.topic_tags.values)).most_common(50))
    vocab = list(top50.keys())
    vocab = vocab[10:]
    vocab_matrix = np.zeros(shape=[len(vocab), len(vocab)])
    vocab2index = dict(zip(vocab, range(len(vocab))))
    index2vocab = dict(zip(range(len(vocab)), vocab))
    for map_tags in topics.topic_tags.values:
        for tag1 in map_tags:
            if tag1 in vocab2index:
                index1 = vocab2index[tag1]
            for tag2 in map_tags:
                if tag2 in vocab2index:
                    index2 = vocab2index[tag2]
                    if index1 != index2:
                        vocab_matrix[index1][index2] += 1
    wordjointwords = defaultdict(set)
    # collect co-ocurrences of <= 15
    for i in range(len(vocab)):
        for j in range(len(vocab)):
            if i != j:
                joint_freq = vocab_matrix[i][j]
                if joint_freq >= 15:
                    word1 = index2vocab[i]
                    word2 = index2vocab[j]
                    wordjointwords[word1].add(word2)
                    wordjointwords[word2].add(word2)
                    wordjointwords[word2].add(word1)
                    wordjointwords[word1].add(word1)
    word2replacement = {}
    # create labels that consist of the tags that co-occur more than 15 times
    for k, v in wordjointwords.items():
        for k2, v2 in wordjointwords.items():
            if k != k2:
                subsest = v.intersection(v2)
                if len(subsest) >= 1:
                    v.update(v2)
                    word2replacement[k] = "/".join(sorted(list(v)))
    replaced_vocab = list(set([word2replacement[w] if w in word2replacement else w for w in vocab]))
    print("reduced vocab is %d" % len(replaced_vocab))
    vocab_matrix = np.zeros(shape=[len(replaced_vocab), len(replaced_vocab)])
    vocab2index = dict(zip(replaced_vocab, range(len(replaced_vocab))))
    for map_tags in topics.topic_tags.values:
        for tag1 in map_tags:
            if tag1 in word2replacement:
                tag1 = word2replacement[tag1]
            if tag1 in vocab2index:
                index1 = vocab2index[tag1]
            for tag2 in map_tags:
                if tag2 in word2replacement:
                    tag2 = word2replacement[tag2]
                if tag2 in vocab2index:
                    index2 = vocab2index[tag2]
                    if index1 != index2:
                        vocab_matrix[index1][index2] += 1
    # plot heat map
    sns.set(font_scale=0.5)
    ax = sns.heatmap(vocab_matrix, xticklabels=replaced_vocab, yticklabels=replaced_vocab)
    plt.tight_layout()
    plt.savefig(outpath, dpi=1500)


def most_frequent_50topics(path_kialo2topics, n):
    """Plot the most frequent topic tags with their frequency"""
    topics = pd.read_csv(path_kialo2topics, sep="\t")
    topics.dropna(inplace=True)
    topic_words = topics['topics'].values
    all_words = []
    for el in topic_words:
        words = el.split(",")
        words = [w.strip() for w in words]
        words = [w.lower() for w in words]
        for w in words:
            all_words.append(w)
    word_frequency = Counter(all_words).most_common(n)
    words = [word for word, _ in word_frequency]
    counts = [counts for _, counts in word_frequency]
    plt.barh(words, counts)
    plt.title("%d most frequent topic tags in kialo" % n)
    plt.ylabel("Frequency")
    plt.xlabel("Words")
    plt.show()


def plot_cosine_similarities(similarities):
    """Plot a histogram of cosine similarities"""
    plt.hist(similarities, density=True, bins=30)  # density=False would make counts
    plt.ylabel('density')
    plt.xlabel('cosine similarity')
    plt.savefig('document_similarity_histogram.png')


def compute_document_similarity(source_maps, target_maps, vectorizer):
    """
    Compute 2 similarity dictionaries between each map in the source maps and each map in the target maps:
    one based on the number of shared vocabulary and one based on the tf-idf document similarity
    :param source_maps: a dictionary with a source map name and the corresponding lemma list
    :param target_maps: a dictionary with a target map name and a corresponding lemma list
    :param vectorizer: a tf-dif vectorizer
    :return: a dictionary with every source map and the shared vocab to every target map, a dictionary with every source map and the tf-idf sim to every target map
    """
    shared_vocab_dic = defaultdict(dict)
    tfidf_dic = defaultdict(dict)
    for source_map_name, source_map_lemmas in source_maps.items():
        pbar = tqdm(total=len(target_maps))
        source_tfidf_vec = get_tfidf_vector(source_map_lemmas, vectorizer)
        for target_map_name, target_map_lemmas in target_maps.items():
            pbar.update(1)
            shared_vocab_dic[source_map_name][target_map_name] = len(shared_vocab(source_map_lemmas, target_map_lemmas))
            target_tfidf_vec = get_tfidf_vector(target_map_lemmas, vectorizer)
            similarity = np.round(tfidf_similairty(source_tfidf_vec, target_tfidf_vec), decimals=2)[0][0]
            tfidf_dic[source_map_name][target_map_name] = similarity
    return shared_vocab_dic, tfidf_dic


def get_most_similar_documents(sim_dic, n):
    """for a given map return the most similar documents (based on the given similarity dictionary)"""
    return sorted(sim_dic.items(), key=operator.itemgetter(1), reverse=True)[:n]


# vocab overlap between each map within deliberatorium
# vocab overlap for each map in delib to the kialo maps
# most important tf-idf content words for each delib map, number of unique content words
if __name__ == '__main__':
    kialo2topics = "/Users/falkne/PycharmProjects/deliberatorium/kialomaps2maintopics.tsv"
    replacements = create_domains(kialo2topics, "data/plots/top50mergeFreq15.png")
    print(replacements)
    # tag_headmap_kialo(kialo2topics, 50, "data/plots/heatmapTop40.png")
    # topic_analysis()
