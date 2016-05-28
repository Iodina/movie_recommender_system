# -*- coding: utf-8 -*-
import numpy as np
from rec_sys import Recommender
import operator


# predictions = [
#             {'i1': 10, 'i2': 4, 'i3': 3, 'i4': 6, 'i5': 10,
#              'i6': 9, 'i7': 6, 'i8': 8, 'i9': 10, 'i10': 8},
#             {'i1': 1, 'i2': 9, 'i3': 8, 'i4': 9, 'i5': 7,
#              'i6': 9, 'i7': 6, 'i8': 9, 'i9': 3, 'i10': 8},
#             {'i1': 10, 'i2': 5, 'i3': 2, 'i4': 7, 'i5': 9,
#              'i6': 8, 'i7': 5, 'i8': 6, 'i9': 7, 'i10': 6}]

# predictions = np.array(([10, 4, 3, 6, 10, 9, 6, 8, 10, 8],
#             [1, 9, 8, 9, 7, 9, 6, 9, 3, 8],
#             [10, 5, 2, 7, 9, 8, 5, 6, 7, 6]))
# print predictions

class GroupRecommender(Recommender):
    def __init__(self, datafile_path=None):
        Recommender.__init__(self, datafile_path=datafile_path)
        self.predictions = None
        self.aggregation_function = {'additive': self.additive,
                                     'multiplicative': self.multiplicative,
                                     'average': self.average,
                                     'average_without_misery': self.average_without_misery,
                                     'least_misery': self.least_misery,
                                     'most_pleasure': self.most_pleasure,
                                     'fairness': self.fairness,
                                     'borda': self.borda,
                                     'copeland': self.copeland,
                                     'plurality_voting': self.plurality_voting,
                                     'approval_voting': self.approval_voting
                                     }

    def _norm(self, vector):
        return np.linalg.norm(vector)

    def _get_top_list(self, res_vector, top=None, relative=True, title=True):
        """
        Determine top `top` films recommended for a Group
        by Group Preferences vector
        :param res_vector: 1D np.array with Group Preferences
        :param top: int number of top movies to recommend
        :param relative: if scores are relative or absolute (weighted)
        :param title: if we need title or not
        :return: [('title' str, score int), (,), (,) ...]
        recommendation for the group
        """
        if not top:
            return res_vector
        prediction = {}
        for index in xrange(len(res_vector)):
            film = self.matrix.indexes_films_map[index]
            prediction[film] = res_vector[index]

        prediction = sorted(prediction.items(), key=operator.itemgetter(1))
        prediction = reversed(prediction[-top:])

        if relative:
            relative_predict = []
            k = 0
            for i in prediction:
                relative_predict.append((i[0], top - k))
                k += 1
            prediction = relative_predict

        if title:
            titled_predict = []
            for i in prediction:
                titled_predict.append((i[0].get_title(), i[1]))
            prediction = titled_predict

        return prediction

    def predict_for_group_merging_recommendations(self, aggregation='additive', **kwargs):
        """
        Return Group Recommendation based on merging individual recommendations
        :param aggregation: aggregation function
        :param kwargs: threshold=3, top=10, weights=[1, 3, 5] and others
        :return:
        """
        self.predictions = self.predicted_rating_submatrix_for_fake()
        aggregate = self.aggregation_function.get(aggregation)
        return aggregate(**kwargs)

    def predict_for_group_merging_profiles(self, **kwargs):
        """
        Return Group Recommendation based on merging profiles
        :param kwargs:
        :return:
        """
        # TODO Implement another method
        # Need to get all fake user's preferences and
        # combine them into one fat fake user and make prediction for him

    def additive(self, **kwargs):
        """Calculate Group preference with additive strategy
        :param top: int
        :return np array in relative values
        """
        top = kwargs.get('top')
        res_vector = self.predictions.sum(axis=0)
        # return self._get_top_list(res_vector, top=top, relative=True)

    def multiplicative(self, **kwargs):
        """Calculate Group preference with multiplicative strategy
        :param top: int
        :return np array in relative values
        """
        top = kwargs.get('top')
        res_vector = self.predictions.prod(axis=0)
        # return self._get_top_list(res_vector, top=top, relative=True)

    def average(self, **kwargs):
        """
        Calculate Group preference with average strategy
        :param top: int
        :return np array in real values
        """
        top = kwargs.get('top')
        res_vector = np.average(self.predictions, axis=0)
        # return self._get_top_list(res_vector, top=top, relative=False)

    def average_without_misery(self, **kwargs):
        """
        Calculate Group preference with average without misery strategy
        :param top: int
        :param threshold: int
        :return np array in real values
        """
        top = kwargs.get('top')
        threshold = kwargs.get('threshold')
        predict = self.predictions
        mask = (predict < threshold)
        idx = mask.any(axis=0)
        predict[:, idx] = 0
        res_vector = np.average(predict, axis=0)
        # return self._get_top_list(res_vector, top=top, relative=False)

    def least_misery(self, **kwargs):
        """
        Calculate Group preference with least misery strategy
        :param kwargs: top: int
        :return: np array in relative values
        """
        top = kwargs.get('top')
        res_vector = np.amin(self.predictions, axis=0)
        # return self._get_top_list(res_vector, top=top, relative=True)

    def most_pleasure(self, **kwargs):
        """
        Calculate Group preference with most pleasure strategy
        :param kwargs: top: int
        :return: np array in relative values
        """
        top = kwargs.get('top')
        res_vector = np.amax(self.predictions, axis=0)
        # return self._get_top_list(res_vector, top=top, relative=True)

    def fairness(self, **kwargs):
        """
        Calculate Group preference with fairness strategy
        :param kwargs: l: int number of top user preferences
        :return: np array in relative values
        """
        l = kwargs.get('l')
        top = kwargs.get('top')
        k = self.predictions.shape[1]
        res_vector = np.zeros((self.predictions.shape[1]))
        while k > 0:
            for u in range(self.predictions.shape[0]):
                list_indexes_top = self.predictions[u].argsort()[-l:][::-1]  # list of indexes of top l scores
                min_items = []  # indexes
                for item_index in list_indexes_top:
                    min_items.append(np.argmin(self.predictions[:, item_index]))
                a = [(self.predictions[i][j], i, j) for i, j in
                     zip(min_items, list_indexes_top) if i != u]
                if a:
                    max_item = max(a)
                    res_vector[max_item[2]] = k
                    self.predictions[:, max_item[2]] = 0
                    k -= 1
                    if k == 0:
                        break
        # return self._get_top_list(res_vector, top=top, relative=True)

    def plurality_voting(self, **kwargs):
        """
        Calculate Group preference with plurality voting strategy
        :param kwargs: l: int number of top user preferences
        :return: np array in relative values
        """
        l = kwargs.get('l')
        top = kwargs.get('top')
        k = self.predictions.shape[1]
        res_vector = np.zeros((self.predictions.shape[1]))
        while k > 0:
            for u in range(self.predictions.shape[0]):
                list_indexes_top = self.predictions[u].argsort()[-l:][::-1]  # list of indexes of top l scores
                sum_items = []  # sum
                for item_index in list_indexes_top:
                    sum_items.append((sum(self.predictions[:, item_index]) - self.predictions[u][item_index], item_index))
                indx_max_sum = max(sum_items)[1]
                res_vector[indx_max_sum] = k
                self.predictions[:, indx_max_sum] = 0
                k -= 1
                if k == 0:
                    break
        # return self._get_top_list(res_vector, top=top, relative=True)

    def borda(self, **kwargs):
        """
        Calculate Group preference with Borda count strategy
        :return:
        """
        top = kwargs.get('top')
        self.predictions = self.predictions.astype(float)
        predict = np.copy(self.predictions)
        for row in range(predict.shape[0]):
            k = 0
            while np.where(predict[row] == 100)[0].shape[0] != predict.shape[1]:
                # find all indexes of minimum element
                arr_indx_min = np.where(predict[row] == predict[row].min())
                num_of_same = arr_indx_min[0].shape[0]
                for j in arr_indx_min[0]:
                    predict[row][j] = 100
                    borda = (((num_of_same + k + 1) * (num_of_same + k) - k * (
                    k + 1)) / 2.) / float(num_of_same) - 1
                    self.predictions[row][j] = borda
                k += num_of_same
        res_vector = self.predictions.sum(axis=0)
        # return self._get_top_list(res_vector, top=top, relative=True)

    def copeland(self, **kwargs):
        """
        Calculate Group preference with Copeland strategy
        :return:
        """
        def wins(k, t):
            B = self.predictions.swapaxes(0, 1)
            return sum(np.greater(B[t], B[k]))

        def looses(k, t):
            B = self.predictions.swapaxes(0, 1)
            return sum(np.less(B[t], B[k]))

        top = kwargs.get('top')
        num_of_items = self.predictions.shape[1]
        A = np.zeros((num_of_items, num_of_items))
        for i in range(num_of_items):
            for j in range(i, num_of_items):
                w = wins(i, j)
                l = looses(i, j)
                if i == j or w == l:
                    A[i, j] = A[j, i] = 0
                elif w > l:
                    A[i, j] = 1
                    A[j, i] = -1
                else:
                    A[i, j] = -1
                    A[j, i] = 1
        res_vector = A.sum(axis=0)
        # return self._get_top_list(res_vector, top=top, relative=True)

    def approval_voting(self, **kwargs):
        """
        Calculate Group preference with approval voting strategy
        :param kwargs: threshold ind
        :return:
        """
        def approved(k, t):
            return self.predictions[k][t] > threshold

        top = kwargs.get('top')
        threshold = kwargs.get('threshold')
        A = np.zeros(self.predictions.shape)
        for i in range(self.predictions.shape[0]):
            for j in range(self.predictions.shape[1]):
                if approved(i, j):
                    A[i, j] = 1
        res_vector = A.sum(axis=0)
        # return self._get_top_list(res_vector, top=top, relative=True)


if __name__ == "__main__":

    import cProfile

    gr = GroupRecommender('test_dataset')
    gr.load_local_data('test_dataset', 100, 0)
    cProfile.run("result = gr.predict_for_group_merging_recommendations(aggregation='additive', threshold=5, top=10, l=3)")
    # result = gr.predict_for_group_merging_recommendations(aggregation='approval_voting', threshold=5, top=10, l=3)
    # for i in result:
    #     print i[0], i[1], "\n"
    # print result
