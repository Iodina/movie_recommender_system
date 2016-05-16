import parser as prs
import numpy as np
import csv

__author__ = 'mishashamen'


class RatingMatrix:
    def __init__(self, users, min_num_of_voted_films=1):
        self.__rating_matrix = None
        self.indexes_users_map = {}
        self.users_indexes_map = {}
        self.indexes_films_map = {}
        self.films_indexes_map = {}

        films = []

        user_index = 0
        for user in users:
            films_with_rate = user.get_films_with_rate()
            if len(films_with_rate) >= min_num_of_voted_films:
                self.indexes_users_map[user_index] = user
                self.users_indexes_map[user] = user_index
                user_index += 1
                films += [fwr['film'] for fwr in films_with_rate]

        film_index = 0
        for film in set(films):
            self.indexes_films_map[film_index] = film
            self.films_indexes_map[film] = film_index
            film_index += 1

            # = np.zeros((user_index, film_index))

    def get_rating_matrix(self):
        if self.__rating_matrix is None:
            self.fill_rating_matrix()
        return self.__rating_matrix

    def fill_rating_matrix(self):
        self.__rating_matrix = np.zeros((len(self.users_indexes_map), len(self.films_indexes_map)))
        for user in self.users_indexes_map.keys():
            films_with_rate = user.get_films_with_rate()
            for film_with_rate in films_with_rate:
                user_index = self.users_indexes_map[user]
                film_index = self.films_indexes_map[film_with_rate['film']]
                self.__rating_matrix[user_index][film_index] = film_with_rate['rate']

    def delete_row(self, index):
        self.__rating_matrix = np.delete(self.__rating_matrix, index, 0)
        user = self.indexes_users_map[index]
        del self.users_indexes_map[user]
        del self.indexes_users_map[index]
        for key, value in self.users_indexes_map.iteritems():
            if value > index:
                self.users_indexes_map[key] = value - 1
                self.indexes_users_map[value - 1] = key
        last_element_index = len(self.users_indexes_map)
        if self.indexes_users_map.get(last_element_index):  # if not last row was deleted
            del self.indexes_users_map[last_element_index]

    def delete_column(self, index):
        self.__rating_matrix = np.delete(self.__rating_matrix, index, 1)
        film = self.indexes_films_map[index]
        del self.films_indexes_map[film]
        del self.indexes_films_map[index]
        for key, value in self.films_indexes_map.iteritems():
            if value > index:
                self.films_indexes_map[key] = value - 1
                self.indexes_films_map[value - 1] = key
        last_element_index = len(self.films_indexes_map)
        if self.indexes_films_map.get(last_element_index):  # if not last column was deleted
            del self.indexes_films_map[last_element_index]

    def save_rating_matrix_as_file(self, filename):
        if self.__rating_matrix is None:
            self.fill_rating_matrix()
        with open(filename, 'wb') as my_file:
            wr = csv.writer(my_file, delimiter=' ')
            for user_index in xrange(self.__rating_matrix.shape[0]):
                for film_index in xrange(self.__rating_matrix.shape[1]):
                    rate = self.__rating_matrix[user_index][film_index]
                    if rate != 0.:
                        wr.writerow([user_index, film_index, rate])



class MatrixCreator:
    rs = prs.RequestSender()
    parser = prs.Parser(rs)

    def create_by_titles_films_users(self, film_names):
        users = []
        for name in film_names:
            film = self.parser.get_film_with_id_by_title(name)
            users += self.parser.get_film_users(film)

        return RatingMatrix(set(users))

if __name__ == "__main__":
    p = prs.Parser(prs.RequestSender())
    f1 = prs.Film(p, 10)
    f2 = prs.Film(p, 20)
    f3 = prs.Film(p, 30)
    f4 = prs.Film(p, 40)

    u1 = prs.User(p, 100)
    u1.add_film_with_rate(f1, 8)
    u1.add_film_with_rate(f2, 7)
    u1.add_film_with_rate(f3, 6)

    u2 = prs.User(p, 200)
    u2.add_film_with_rate(f1, 9)
    u2.add_film_with_rate(f3, 8)
    u2.add_film_with_rate(f4, 3)

    u3 = prs.User(p, 300)
    u3.add_film_with_rate(f2, 1)
    u3.add_film_with_rate(f3, 2)

    u4 = prs.User(p, 400)
    u4.add_film_with_rate(f3, 5)

    rm = RatingMatrix(set([u1, u2, u3, u4]))

    print [str(key) + ' : ' + str(rm.indexes_users_map[key].get_id()) for key in rm.indexes_users_map]
    print [str(key) + ' : ' + str(rm.indexes_films_map[key].get_id()) for key in rm.indexes_films_map]

    print rm.get_rating_matrix()
    rm.delete_row(0)
    rm.delete_column(0)
    print rm.get_rating_matrix()
    print [str(key) + ' : ' + str(rm.indexes_users_map[key].get_id()) for key in rm.indexes_users_map]
    print [str(key.get_id()) + ' : ' + str(rm.users_indexes_map[key]) for key in rm.users_indexes_map]
    print [str(key) + ' : ' + str(rm.indexes_films_map[key].get_id()) for key in rm.indexes_films_map]
