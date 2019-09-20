"""
data process
"""

import os
import json
import random
import copy
from collections import Counter
from itertools import chain
from typing import Dict, Tuple, Optional, List, Union

import gensim
import numpy as np


class RelationData(object):
    def __init__(self, output_path: str, sequence_length: int = 200, num_classes: int = 3, num_support: int = 5,
                 num_queries: int = 50, num_tasks: int = 1000, num_eval_tasks: int = 100,
                 stop_word_path: Optional[str] = None,
                 embedding_size: Optional[int] = None, low_freq: int = 5,
                 word_vector_path: Optional[str] = None, is_training: bool = True):
        """
        init method
        :param output_path: path of train/eval data
        :param num_classes: number of support class
        :param num_support: number of support sample per class
        :param num_queries: number of query sample per class
        :param num_tasks: number of pre-sampling tasks, this will speeding up train
        :param num_eval_tasks: number of pre-sampling tasks in eval stage
        :param stop_word_path: path of stop word file
        :param embedding_size: embedding size
        :param low_freq: frequency of words
        :param word_vector_path: path of word vector file(eg. word2vec, glove)
        :param is_training: bool
        """

        self.__output_path = output_path
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)

        self.__sequence_length = sequence_length
        self.__num_classes = num_classes
        self.__num_support = num_support
        self.__num_queries = num_queries
        self.__num_tasks = num_tasks
        self.__num_eval_tasks = num_eval_tasks
        self.__stop_word_path = stop_word_path
        self.__embedding_size = embedding_size
        self.__low_freq = low_freq
        self.__word_vector_path = word_vector_path
        self.__is_training = is_training

        self.vocab_size = None
        self.word_vectors = None
        self.current_category_index = 0  # record current sample category

    @staticmethod
    def load_data(data_path: str) -> Dict[str, Dict[str, List[List[str]]]]:
        """
        read train/eval data
        :param data_path:
        :return: dict. {class_name: {sentiment: [[]], }, ...}
        """
        category_dirs = os.listdir(data_path)
        categories_data = {}
        # foreach all product review dir
        for category_dir in category_dirs:
            file_dirs = os.path.join(data_path, category_dir)
            # fetch five sentiment reviews for each product
            sentiment_files = os.listdir(file_dirs)
            sentiment_data = {}  # save review for each product, [[], [], [], [], []]
            for file in sentiment_files:
                prefix = os.path.splitext(file)[0]
                file_path = os.path.join(file_dirs, file)
                with open(file_path, "r") as fr:
                    data = [line.strip().split() for line in fr.readlines()]
                sentiment_data[prefix] = data
            categories_data[category_dir] = sentiment_data

        return categories_data

    def remove_stop_word(self, data: Dict[str, Dict[str, List[List[str]]]]) -> List[str]:
        """
        remove low frequency words and stop words, construct vocab
        :param data: {class_name: {sentiment: [[]], }, ...}
        :return:
        """
        all_words = []
        for category, category_data in data.items():
            for sentiment, sentiment_data in category_data.items():
                all_words.extend(list(chain(*sentiment_data)))
        word_count = Counter(all_words)  # statistic the frequency of words
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # remove low frequency word
        words = [item[0] for item in sort_word_count if item[1] > self.__low_freq]

        # if stop word file exists, then remove stop words
        if self.__stop_word_path:
            with open(self.__stop_word_path, "r", encoding="utf8") as fr:
                stop_words = [line.strip() for line in fr.readlines()]
            words = [word for word in words if word not in stop_words]

        return words

    def get_word_vectors(self, vocab: List[str]) -> np.ndarray:
        """
        load word vector file,
        :param vocab: vocab
        :return:
        """
        pad_vector = np.zeros(self.__embedding_size)  # set the "<PAD>" vector to 0
        word_vectors = (1 / np.sqrt(len(vocab) - 1) * (2 * np.random.rand(len(vocab) - 1, self.__embedding_size) - 1))
        if os.path.splitext(self.__word_vector_path)[-1] == ".bin":
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self.__word_vector_path, binary=True)
        else:
            word_vec = gensim.models.KeyedVectors.load_word2vec_format(self.__word_vector_path)

        for i in range(1, len(vocab)):
            try:
                vector = word_vec.wv[vocab[i]]
                word_vectors[i, :] = vector
            except:
                print(vocab[i] + "not exist word vector file")
        word_vectors = np.vstack((pad_vector, word_vectors))
        return word_vectors

    def gen_vocab(self, words: List[str]) -> Dict[str, int]:
        """
        generate word_to_index mapping table
        :param words:
        :return:
        """
        if self.__is_training:
            vocab = ["<PAD>", "<UNK>"] + words

            self.vocab_size = len(vocab)

            if self.__word_vector_path:
                word_vectors = self.get_word_vectors(vocab)
                self.word_vectors = word_vectors
                # save word vector to npy file
                np.save(os.path.join(self.__output_path, "word_vectors.npy"), self.word_vectors)

            word_to_index = dict(zip(vocab, list(range(len(vocab)))))

            # save word_to_index to json file
            with open(os.path.join(self.__output_path, "word_to_index.json"), "w") as f:
                json.dump(word_to_index, f)
        else:
            with open(os.path.join(self.__output_path, "word_to_index.json"), "r") as f:
                word_to_index = json.load(f)

        return word_to_index

    @staticmethod
    def trans_to_index(data: Dict[str, Dict[str, List[List[str]]]], word_to_index: Dict[str, int]) -> \
            Dict[str, Dict[str, List[List[int]]]]:
        """
        transformer token to id
        :param data:
        :param word_to_index:
        :return: {class_name: [[], [], ], ..}
        """
        data_ids = {category: {sentiment: [[word_to_index.get(token, word_to_index["<UNK>"]) for token in line]
                                           for line in sentiment_data]
                               for sentiment, sentiment_data in category_data.items()}
                    for category, category_data in data.items()}
        return data_ids

    def choice_support_query(self, category_data: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        selecting support data and query data form all data for a category
        :param category_data: all data for a category
        :return:
        """
        support_data = random.sample(category_data, self.__num_support)
        other_data = copy.copy(category_data)
        [other_data.remove(data) for data in support_data]
        query_data = random.sample(other_data, self.__num_queries)

        # padding
        support_data = self.padding(support_data)
        query_data = self.padding(query_data)
        return support_data, query_data

    def samples(self, data_ids: Dict[str, Dict[str, List[List[int]]]]) \
            -> List[Dict[str, Union[List[List[List[int]]], List[List[int]], List[int]]]]:
        """
        positive and negative sample from raw data
        :param data_ids:
        :return:
        """
        if self.__is_training:
            label_to_index = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
            with open(os.path.join(self.__output_path, "label_to_index.json"), "w") as f:
                json.dump(label_to_index, f)

        # product name list
        category_list = list(data_ids.keys())

        tasks = []
        if self.__is_training:
            num_tasks = self.__num_tasks
        else:
            num_tasks = self.__num_eval_tasks
        for i in range(num_tasks):
            # randomly choice a category to construct train sample
            support_category = random.choice(category_list)
            support_set = []  # [num_classes, num_support, sequence_length]
            query_set = []  # [num_classes * num_queries, sequence_length]
            labels = []
            for idx, sentiment in data_ids[support_category].items():
                support_data, query_data = self.choice_support_query(sentiment)
                label = [int(idx) - 1] * len(query_data)
                support_set.append(support_data)
                query_set.extend(query_data)
                labels.extend(label)
            tasks.append(dict(support=support_set, queries=query_set, labels=labels))
        return tasks

    def gen_data(self, file_path: str) -> Dict[str, Dict[str, List[List[int]]]]:
        """
        Generate data that is eventually input to the model
        :return:
        """
        # load data
        data = self.load_data(file_path)
        # remove stop word
        words = self.remove_stop_word(data)
        word_to_index = self.gen_vocab(words)

        data_ids = self.trans_to_index(data, word_to_index)
        return data_ids

    def padding(self, sentences: List[List[int]]) -> List[List[int]]:
        """
        padding according to the predefined sequence length
        :param sentences:
        :return:
        """
        sentence_pad = [sentence[:self.__sequence_length] if len(sentence) > self.__sequence_length
                        else sentence + [0] * (self.__sequence_length - len(sentence))
                        for sentence in sentences]
        return sentence_pad

    def next_batch(self, data_ids: Dict[str, Dict[str, List[List[int]]]]) \
            -> Dict[str, Union[List[List[List[int]]], List[List[int]], List[int]]]:
        """
        train a task at every turn
        :param data_ids:
        :return:
        """
        tasks = self.samples(data_ids)

        for task in tasks:
            yield task
