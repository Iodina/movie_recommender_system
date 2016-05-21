import parser as prs
import numpy as np
import csv

__author__ = 'mishashamen'


class RatingMatrix:
    def __init__(self, users=[], MIN_NUM_OF_VOTED_FILMS=1):
        """
        Builds rating matrix (`A`) and adds fake users with
        films and rates from `users`
        Mapping index <-> User and index <-> Film
        helps to find object by matrix index
        :param users: [{Film: rate, ...}, ...]
        :param MIN_NUM_OF_VOTED_FILMS:
        :return:
        """
        self.indexes_with_fake_user_ids = {}  # {int(user_index): int(fake_user_id)}
        self.rating_matrix = None   # numPy array
        self.indexes_users_map = {}  # {int(index): User}
        self.users_indexes_map = {}  # {User: int(index)}
        self.indexes_films_map = {}  # {int(index): Film}
        self.films_indexes_map = {}  # {Film: int(index)}

        films = []

        user_index = 0
        for user in users:
            print 'user id:' + str(user.get_id())
            films_with_rate = user.get_films_with_rate()
            if films_with_rate is None:
                pass
            elif len(films_with_rate) >= MIN_NUM_OF_VOTED_FILMS:
                self.indexes_users_map[user_index] = user
                self.users_indexes_map[user] = user_index
                if user.get_id() < 0:
                    self.indexes_with_fake_user_ids[user_index] = user.get_id()
                user_index += 1
                films += [fwr['film'] for fwr in films_with_rate]

        film_index = 0
        for film in set(films):
            print 'film id:' + str(film.get_id())
            self.indexes_films_map[film_index] = film
            self.films_indexes_map[film] = film_index
            film_index += 1

    def get_rating_matrix(self):
        if self.rating_matrix is None:
            self.fill_rating_matrix()
        return self.rating_matrix

    def fill_rating_matrix(self):
        self.rating_matrix = np.zeros((len(self.users_indexes_map), len(self.films_indexes_map)))
        for user in self.users_indexes_map.keys():
            films_with_rate = user.get_films_with_rate()
            for film_with_rate in films_with_rate:
                user_index = self.users_indexes_map[user]
                film_index = self.films_indexes_map[film_with_rate['film']]
                self.rating_matrix[user_index][film_index] = film_with_rate['rate']

    def delete_row(self, index):
        if self.rating_matrix is not None:
            self.rating_matrix = np.delete(self.rating_matrix, index, 0)

        if index in self.indexes_with_fake_user_ids.keys():
            del self.indexes_with_fake_user_ids[index]

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
        if self.rating_matrix is not None:
            self.rating_matrix = np.delete(self.rating_matrix, index, 1)
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
        if self.rating_matrix is None:
            self.fill_rating_matrix()
        with open(filename, 'wb') as my_file:
            wr = csv.writer(my_file, delimiter=' ')
            for user_index in xrange(self.rating_matrix.shape[0]):
                for film_index in xrange(self.rating_matrix.shape[1]):
                    rate = self.rating_matrix[user_index][film_index]
                    if rate != 0.:
                        wr.writerow([user_index, film_index, rate])

        with open(filename+'_user_map', 'wb') as my_file:
            wr = csv.writer(my_file, delimiter=' ')
            for key, value in self.indexes_users_map.iteritems():
                wr.writerow([key, value.get_id()])

        with open(filename+'_film_map', 'wb') as my_file:
            wr = csv.writer(my_file, delimiter=' ')
            for key, value in self.indexes_films_map.iteritems():
                wr.writerow([key, value.get_id()])


class MatrixCreator:
    def __init__(self, MAX_COUNT_USER_FILMS = None, MAX_COUNT_FILM_USERS = None):
        self.parser = prs.Parser(MAX_COUNT_USER_FILMS, MAX_COUNT_FILM_USERS)

    def create_matrix_by_film_titles(self, film_names_with_rate_list):   # [{film_name : film_rate, ...}, ...]
        users = []
        fake_user_id = -1
        for film_names_with_rate in film_names_with_rate_list:
            fake_user = prs.User(self.parser, fake_user_id)
            for film_name, rate in film_names_with_rate.iteritems():
                film = self.parser.get_film_with_id_by_title(film_name)
                print film_name + ', id: ' + str(film.get_id())
                fake_user.add_film_with_rate(film, rate)
                users += self.parser.get_film_users(film)
            users.append(fake_user)
            fake_user_id -= 1

        return RatingMatrix(set(users))

    def restore_from_file(self, file_name):
        rm = RatingMatrix()
        with open(file_name+'_user_map', 'rb') as my_file:
            r = csv.reader(my_file)
            for row in r:
                index, id = row[0].split(' ')
                rm.indexes_users_map[int(index)] = prs.User(self.parser, id)
                rm.users_indexes_map[rm.indexes_users_map[int(index)]] = int(index)
                if int(id) < 0:
                    rm.indexes_with_fake_user_ids[int(index)] = int(id)

        with open(file_name+'_film_map', 'rb') as my_file:
            r = csv.reader(my_file)
            for row in r:
                index, id = row[0].split(' ')
                rm.indexes_films_map[int(index)] = prs.Film(self.parser, id)
                rm.films_indexes_map[rm.indexes_films_map[int(index)]] = int(index)

        rm.rating_matrix = np.zeros((len(rm.users_indexes_map), len(rm.films_indexes_map)))

        with open(file_name, 'rb') as my_file:
            r = csv.reader(my_file)
            for row in r:
                user_index, film_index, rate = row[0].split(' ')
                rm.rating_matrix[int(user_index)][int(film_index)] = float(rate)

                user = rm.indexes_users_map[int(user_index)]
                film = rm.indexes_films_map[int(film_index)]
                user.add_film_with_rate(film, float(rate))

        return rm

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
    print rm.get_rating_matrix()
