#!/usr/bin/env python
# -*- coding: utf-8 -*-
from recsys.algorithm.factorize import SVD
import rating_matrix as rm
import operator

__author__ = 'mishashamen'

class Recommender:
    def __init__(self):
        self.svd = SVD()
        self.matrix = None
        self.datafile_path = None

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
        np_matrix = self.matrix.get_rate_matrix()
        for index in xrange(np_matrix.shape[1]):
            rate = self.svd.predict(user_index, index,
                                    MIN_VALUE=min_rate,
                                    MAX_VALUE=max_rate)
            film = self.matrix.indexes_films_map[index]
            prediction[film] = rate

        if top:
            prediction = sorted(prediction.items(), key=operator.itemgetter(1))
            prediction = prediction[-top:]

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
    recommender.load_web_data('dataset',
                              [{"Легенда": 9, 
                                "Пираты Карибского моря": 10, 
                                "Дневники вампира": 5,
                                "Зверополис": 9,
                                "Эпоха дракона: Искупление": 6,
                                "Запах женщины": 8},
                                {"Эпоха дракона: Искупление": 3,
                                "Машинист": 8, 
                                "Зверополис": 8, 
                                "Илья Муромец":10, 
                                "Иван Васильевич меняет профессию": 10, 
                                "Эверест": 6 },
                                {"Запах женщины": 10,
                                "Мальчики-налетчики": 2,
                                "Тренер Картер": 8,
                                "Заплати другому": 10,
                                "Живая сталь": 9,
                                "Одержимость": 5},
                                {"12 разгневанных мужчин": 10,
                                "Омерзительная восьмерка": 6,
                                "Гран Торино": 8,
                                "Игры разума": 9,
                                "В джазе только девушки": 10,
                                "Пролетая над гнездом кукушки": 9,
                                "Одержимость": 9,
                                "Назад в будущее": 8,
                                "Титаник": 7},
                                {"Жестокие игры": 10,
                                "Двое: Я и моя тень": 9,
                                "Омерзительная восьмерка": 8,
                                "Зверополис": 10,
                                "Игры разума": 9,
                                "На гребне волны": 2, 
                                "Пролетая над гнездом кукушки": 5,
                                "Титаник": 9}
                                ],
                              100, 0)

    # for film, rate in output_result(rate_matrix, 'scent_of_women').iteritems():
    #     if rate > 9:
    #         print film.get_title()
