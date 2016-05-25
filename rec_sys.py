#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
from recsys.algorithm.factorize import SVD
import rating_matrix as rm
import operator
import numpy as np
import csv
from shutil import copyfile

__author__ = 'mishashamen'

class Recommender:
    def __init__(self, datafile_path=None):
        self.svd = SVD()
        self.matrix = None
        self.datafile_path = datafile_path

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

    def filter_films_data(self, min_user_votes):
        film_indexes = []
        counter = collections.Counter()
        with open(self.datafile_path, 'rb') as my_file:
            r = csv.reader(my_file)
            for row in r:
                user_index, film_index, rate = row[0].split(' ')
                counter[int(film_index)] += 1

            for k, v in counter.iteritems():
                if v < min_user_votes:
                    film_indexes.append(k)

        copyfile(self.datafile_path+'_user_map', self.datafile_path+'_'+str(min_user_votes)+'_user_map')

        new_indexes = {}
        with open(self.datafile_path+'_film_map', 'rb') as read_file:
            r = csv.reader(read_file)
            with open(self.datafile_path+'_'+str(min_user_votes)+'_film_map', 'wb') as write_file:
                wr = csv.writer(write_file, delimiter=' ')
                index = 0
                for row in r:
                    film_index, film_id = row[0].split(' ')
                    if int(film_index) in film_indexes:
                        continue
                    new_indexes[film_index] = index
                    wr.writerow([index, film_id])
                    index += 1

        with open(self.datafile_path, 'rb') as read_file:
            r = csv.reader(read_file)
            with open(self.datafile_path+'_'+str(min_user_votes), 'wb') as write_file:
                wr = csv.writer(write_file, delimiter=' ')
                for row in r:
                    user_index, film_index, rate = row[0].split(' ')
                    if int(film_index) in film_indexes:
                        continue
                    wr.writerow([user_index, new_indexes[film_index], rate])


if __name__ == "__main__":
    r = Recommender('dataset')
    # r.load_local_data('dataset', 100, 0)
    # print len(r.matrix.find_films_to_filter(2))
    # print r.matrix.rating_matrix.shape[1]
    r.filter_films_data(15)
    # r.load_local_data('small_dataset_2', 100, 0)
    # print r.matrix.rating_matrix.shape[1]



