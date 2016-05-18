#!/usr/bin/env python
# -*- coding: utf-8 -*-
from recsys.algorithm.factorize import SVD
import rating_matrix as rm

__author__ = 'mishashamen'

if __name__ == "__main__":
    # rate_matrix = rm.MatrixCreator(MAX_COUNT_FILM_USERS=20, MAX_COUNT_USER_FILMS=20).create_users_by_film_titles([{
    #     "Запах женщины": 10}])
    # rate_matrix.save_rating_matrix_as_file('scent_of_women')

    rate_matrix = rm.MatrixCreator().restore_from_file('scent_of_women')
    fake_user_index = rate_matrix.indexes_with_fake_user_ids.keys()[0]
    fake_user = rate_matrix.indexes_users_map[fake_user_index]
    fake_user_film = fake_user.get_films_with_rate()[0]['film']  # Scent of women
    film_index = rate_matrix.films_indexes_map[fake_user_film]

    print fake_user_film.get_title()
    # print rate_matrix.get_rating_matrix()
    svd = SVD()
    svd.load_data(filename='scent_of_women', sep=' ', format={'col': 1, 'row': 0, 'value': 2, 'ids': int})
    K = 100
    svd.compute(k=K, min_values=0, pre_normalize=None, mean_center=True, post_normalize=True)
    print svd.predict(fake_user_index, film_index, MIN_VALUE=1.0, MAX_VALUE=10.0)
