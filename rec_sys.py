#!/usr/bin/env python
# -*- coding: utf-8 -*-
from recsys.algorithm.factorize import SVD
import rating_matrix as rm

__author__ = 'mishashamen'

if __name__ == "__main__":
    rate_matrix = rm.MatrixCreator(MAX_COUNT_FILM_USERS=20, MAX_COUNT_USER_FILMS=20). \
        create_by_titles_films_users(['Запах женщины'])
    rate_matrix.save_rating_matrix_as_file('scent.dat')

    svd = SVD()
    svd.load_data(filename='scent.dat', sep=' ', format={'col': 1, 'row': 0, 'value': 2, 'ids': int})
    K = 100
    svd.compute(k=K, min_values=0, pre_normalize=None, mean_center=True, post_normalize=True)
    print svd.predict(3, 3, MIN_VALUE=1.0, MAX_VALUE=10.0)
