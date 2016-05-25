#!/usr/bin/env python
# -*- coding: utf-8 -*-
from recsys.algorithm.factorize import SVD
import rating_matrix as rm
import operator
import numpy as np

__author__ = 'mishashamen'

class Recommender:
    def __init__(self):
        self.svd = SVD()
        self.matrix = None
        self.datafile_path = None
        self.predict_matrix = None

    def load_web_data(self, filename, film_names_with_rate_list, K, min_values,
                  MAX_COUNT_USER_FILMS=None, MAX_COUNT_FILM_USERS=None):
        self.matrix = rm.MatrixCreator(MAX_COUNT_USER_FILMS, MAX_COUNT_FILM_USERS).\
            create_matrix_by_film_titles(film_names_with_rate_list)
        self.matrix.save_rating_matrix_as_file(filename)
        self.datafile_path = filename
        self.__compute_matrix(K, min_values)

    def load_local_data(self, filename, K, min_values):
        self.matrix = rm.MatrixCreator().restore_from_file(filename)
        self.datafile_path = filename
        self.__compute_matrix(K, min_values)

    def get_predictions_for_all_users(self, min_rate=1, max_rate=10, top = None, K=None, min_values=0):
        if K:
            self.__compute_matrix(K)

        self.predict_matrix = np.empty([len(self.matrix.users_indexes_map), len(self.matrix.films_indexes_map)])
        for user in self.matrix.users_indexes_map.keys():
            for film in self.matrix.films_indexes_map.keys():
                user_index = self.matrix.users_indexes_map[user]
                film_index = self.matrix.films_indexes_map[film]
                self.predict_matrix[user_index][film_index] = self.svd.predict(user_index, film_index, MIN_VALUE=min_rate, MAX_VALUE=max_rate)
        return self.predict_matrix


    def predict_for_user(self, user_index, min_rate=1, max_rate=10, top = None, K=None, min_values=None):
        """
        :param K: to change the number of properties
        :return: {Film : int(rate), ...} or
                [(Film, int(rate)), ...] if top is not None
        """
        if K:
            self.__compute_matrix(K)

        prediction = {}
        np_matrix = self.matrix.get_rating_matrix()
        for index in xrange(np_matrix.shape[1]):
            rate = self.svd.predict(user_index, index,
                                    MIN_VALUE=min_rate,
                                    MAX_VALUE=max_rate)
            film = self.matrix.indexes_films_map[index]
            prediction[film] = rate

        if top:
            prediction = sorted(prediction.items(), key=operator.itemgetter(1))
            prediction = list(reversed(prediction[-top:]))

        return prediction

    def predict_for_all_fake_users(self, min_rate=1, max_rate=10, top = None, K=None, min_values=0):
        """
        :param K: to change the number of properties
        :return: [{Film : int(rate), ...}, ...]
        """
        if K:
            self.__compute_matrix(K)

        predictions = []

        for user_index in self.matrix.indexes_with_fake_user_ids.keys():
            prediction = self.predict_for_user(user_index, min_rate, max_rate, top)
            predictions.append(prediction)

        return predictions

    def __compute_matrix(self, K,
                         min_values=0,
                         pre_normalize=None,
                         mean_center=True,
                         post_normalize=True):
        self.svd.load_data(self.datafile_path, sep=' ', format={'col': 1, 'row': 0, 'value': 2, 'ids': int})
        self.svd.compute(K, min_values, pre_normalize, mean_center, post_normalize, savefile=None)



if __name__ == "__main__":
    recommender = Recommender()
    # recommender.load_web_data('dataset',
    #                           [{'Запах женщины': 9, 'The Usual Suspects': 8, 'The Departed': 8,
    #                             'Тутси': 7, 'Выпускник': 10, 'Залечь на дно в Брюгге': 4, 'Евротур': 7,
    #                             'Goodfellas': 6, 'Донни Браско': 8, 'Амели': 3, 'Идиократия': 7}],
    #                           100, 0, 10, 10)

    recommender.load_local_data('dataset', K=100, min_values=0)
    m = recommender.matrix.get_rating_matrix()

    m1 = recommender.get_predictions_for_all_users()


