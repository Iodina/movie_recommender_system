# -*- coding: utf-8 -*-
import numpy as np
from rec_sys import Recommender
import operator
import cProfile
import math
import xlwt

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
    def __init__(self, datafile_path):
        Recommender.__init__(self, datafile_path=datafile_path)
        self.load_local_data(datafile_path, 100, 0)
        self.predictions = self.predicted_rating_submatrix_for_fake()
        # self.predictions = np.array(([5, 4, 3, 1],
        #                             [2, 3, 4, 5],
        #                             [2, 1, 1, 3]))
        self.initial_rating = self.initial_rating_submatrix()
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

    def _get_top_list(self, res_vector, top=None, relative=True, title=False):
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

            # return np.sort(res_vector)
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
        # self.predictions = self.predicted_rating_submatrix_for_fake()
        aggregate = self.aggregation_function.get(aggregation)
        return aggregate(**kwargs)

    def predict_for_group_merging_profiles(self, **kwargs):
        """
        Return Group Recommendation based on merging profiles
        :param kwargs:
        :return:
        """
        top = kwargs.get('top')
        # initial_rating = self.initial_rating_submatrix()
        aggregated_user = np.zeros((self.initial_rating.shape[1],))
        for j in range(self.initial_rating.shape[1]):
            number = np.count_nonzero(self.initial_rating[:, j])
            if number:
                aggregated_user[j] = float(sum(self.initial_rating[:, j])) / number
            else:
                aggregated_user[j] = 0.
        indx = self.matrix.add_row(aggregated_user, self.datafile_path)
        res_vector = self.predicted_rating_submatrix([indx])[0]
        return self._get_top_list(res_vector, top=top, relative=False)

    def additive(self, **kwargs):
        """Calculate Group preference with additive strategy
        :param top: int
        :return np array in relative values
        """
        top = kwargs.get('top')
        res_vector = self.predictions.sum(axis=0)
        return self._get_top_list(res_vector, top=top, relative=True)

    def multiplicative(self, **kwargs):
        """Calculate Group preference with multiplicative strategy
        :param top: int
        :return np array in relative values
        """
        top = kwargs.get('top')
        res_vector = self.predictions.prod(axis=0)
        return self._get_top_list(res_vector, top=top, relative=True)

    def average(self, **kwargs):
        """
        Calculate Group preference with average strategy
        :param top: int
        :return np array in real values
        """
        top = kwargs.get('top')
        res_vector = np.average(self.predictions, axis=0)
        return self._get_top_list(res_vector, top=top, relative=False)

    def average_without_misery(self, **kwargs):
        """
        Calculate Group preference with average without misery strategy
        :param top: int
        :param threshold: int
        :return np array in real values
        """
        top = kwargs.get('top')
        threshold = kwargs.get('threshold')
        predict = np.copy(self.predictions)
        mask = (predict < threshold)
        idx = mask.any(axis=0)
        predict[:, idx] = 0
        res_vector = np.average(predict, axis=0)
        return self._get_top_list(res_vector, top=top, relative=False)

    def least_misery(self, **kwargs):
        """
        Calculate Group preference with least misery strategy
        :param kwargs: top: int
        :return: np array in relative values
        """
        top = kwargs.get('top')
        res_vector = np.amin(self.predictions, axis=0)
        return self._get_top_list(res_vector, top=top, relative=True)

    def most_pleasure(self, **kwargs):
        """
        Calculate Group preference with most pleasure strategy
        :param kwargs: top: int
        :return: np array in relative values
        """
        top = kwargs.get('top')
        res_vector = np.amax(self.predictions, axis=0)
        return self._get_top_list(res_vector, top=top, relative=True)

    # all unique
    def fairness(self, **kwargs):
        """
        Calculate Group preference with fairness strategy
        :param kwargs: l: int number of top user preferences
        :return: np array in relative values
        """
        l = kwargs.get('l')
        top = kwargs.get('top')
        predictions = np.copy(self.predictions)
        k = predictions.shape[1]
        res_vector = np.zeros((predictions.shape[1]))
        while k > 0:
            for u in range(predictions.shape[0]):
                list_indexes_top = predictions[u].argsort()[-l:][::-1]  # list of indexes of top l scores
                min_items = []  # indexes
                for item_index in list_indexes_top:
                    min_items.append(np.argmin(predictions[:, item_index]))
                a = [(predictions[i][j], i, j) for i, j in
                     zip(min_items, list_indexes_top) if i != u]
                if a:
                    max_item = max(a)
                    res_vector[max_item[2]] = k
                    predictions[:, max_item[2]] = 0
                    k -= 1
                    if k == 0:
                        break
        return self._get_top_list(res_vector, top=top, relative=True)

    # all unique
    def plurality_voting(self, **kwargs):
        """
        Calculate Group preference with plurality voting strategy
        :param kwargs: l: int number of top user preferences
        :return: np array in relative values
        """
        l = kwargs.get('l')
        top = kwargs.get('top')
        predictions = np.copy(self.predictions)
        k = predictions.shape[1]
        res_vector = np.zeros((predictions.shape[1]))
        while k > 0:
            for u in range(predictions.shape[0]):
                list_indexes_top = predictions[u].argsort()[-l:][::-1]  # list of indexes of top l scores
                sum_items = []  # sum
                for item_index in list_indexes_top:
                    sum_items.append((sum(predictions[:, item_index]) - predictions[u][item_index], item_index))
                indx_max_sum = max(sum_items)[1]
                res_vector[indx_max_sum] = k
                predictions[:, indx_max_sum] = 0
                k -= 1
                if k == 0:
                    break
        return self._get_top_list(res_vector, top=top, relative=True)

    def borda(self, **kwargs):
        """
        Calculate Group preference with Borda count strategy
        :return:
        """
        top = kwargs.get('top')
        predictions = np.copy(self.predictions.astype(float))
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
                    predictions[row][j] = borda
                k += num_of_same
        res_vector = predictions.sum(axis=0)
        return self._get_top_list(res_vector, top=top, relative=True)

    def copeland(self, **kwargs):
        """
        Calculate Group preference with Copeland strategy
        :return:
        """
        def wins(k, t):
            B = predictions.swapaxes(0, 1)
            return sum(np.greater(B[t], B[k]))

        def looses(k, t):
            B = predictions.swapaxes(0, 1)
            return sum(np.less(B[t], B[k]))

        top = kwargs.get('top')
        predictions = np.copy(self.predictions)
        num_of_items = predictions.shape[1]
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
        return self._get_top_list(res_vector, top=top, relative=True)

    # TODO Need to check
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
        res_vector_1 = A.sum(axis=0)
        res_vector = np.zeros((self.predictions.shape[1],))
        # modification
        for indx in range(self.predictions.shape[1]):
            res_vector[indx] = res_vector_1[indx] * self.predictions.sum(axis=0)[indx]
        return self._get_top_list(res_vector, top=top, relative=True)

    def initial_rating_submatrix(self):
        fake_list = self.matrix.indexes_with_fake_user_ids.keys()
        i_size = len(fake_list)
        j_size = self.matrix.rating_matrix.shape[1]
        initial = np.zeros((i_size, j_size))
        for i in range(i_size):
            initial[i, :] = self.matrix.rating_matrix[fake_list[i]]
        return initial

    def evaluate(self, aggregation="additive", l=3, threshold=5, method='after'):
        """
        Return evaluation function value
        :param aggregation:
        :return:
        """
        def get_predicted_rating_minima(j, rating):
            """
            Get minimum rank with similar scores
            :param j:
            :return:
            """
            sorted_indexes = np.argsort(rating)[::-1]
            sorted_scores = np.sort(rating)[::-1]
            rank = np.where(sorted_indexes == j)[0][0]
            score = sorted_scores[rank]
            if score:
                return min(np.where(sorted_scores == score)[0])
            else:
                return None

        # self.predictions = self.predicted_rating_submatrix_for_fake()
        # initial_rating = self.initial_rating_submatrix()
        if method == 'after':
            aggregate = self.aggregation_function.get(aggregation)
            self.rating = aggregate(threshold=threshold, l=l)
        elif method == 'before':
            self.rating = self.predict_for_group_merging_profiles()
        num_users = self.predictions.shape[0]
        num_items = self.predictions.shape[1]
        aggregate_sum = []
        for i in range(num_users):
            num_of_nonzero = np.nonzero(self.initial_rating[i])[0].shape[0]
            mean = np.sum(self.initial_rating[i]) / float(num_of_nonzero)
            summa = 0
            for j in range(num_items):
                if self.initial_rating[i][j]:
                    rank_j = get_predicted_rating_minima(j, self.rating)
                    if rank_j is not None:
                        summa += float(self.initial_rating[i][j] - mean) / math.pow(2, float(rank_j)/num_items)
            aggregate_sum.append(summa)
        eval_mean = float(sum(aggregate_sum)) / num_users
        eval_misery = min(aggregate_sum)
        return eval_mean, eval_misery

    def MAE(self, GROUND_TRUTH, TEST, t=None):
        res_sum = 0.0
        not_zero_count = 0
        for i in range(0, len(GROUND_TRUTH)):
            r = GROUND_TRUTH[i]
            r_pred = TEST[i]
            # if ((t is None) and r > 0.) or ((t is not None) and r >= t):
            if r_pred > 0 and r > 0:
                not_zero_count += 1
                res_sum += abs(r - r_pred)
        return res_sum / not_zero_count

    def RMSE(self, GROUND_TRUTH, TEST, t=None):
        res_sum = 0.0
        not_zero_count = 0
        for i in range(0, len(GROUND_TRUTH)):
            r = GROUND_TRUTH[i]
            r_pred = TEST[i]

            # if ((t is None) and r > 0.) or ((t is not None) and r >= t):
            if r_pred > 0 and r > 0:
                not_zero_count += 1
                res_sum += abs(r - r_pred) * abs(r - r_pred)
        return math.sqrt(res_sum) / not_zero_count

    def evaluate_aggregation_before(self, t=None):
        maes = []
        rmses = []

        before = self.initial_rating

        # can be not defined
        after_aggregated = self.rating

        N = before.shape[0]
        for count in xrange(N):
            try:
                maes.append(self.MAE(before[count], after_aggregated, t))
                rmses.append(self.RMSE(before[count], after_aggregated, t))
            except Exception:
                pass

        return sum(maes) / N, sum(rmses) / N, max(maes), max(rmses)
        return sum(maes) / N, sum(rmses) / N, max(maes), max(rmses)

    def evaluate_aggregation_after(self, aggr, t=None):
        maes = []
        rmses = []

        before = self.initial_rating
        after_aggregated = self.predict_for_group_merging_recommendations(aggregation=aggr, threshold=t)

        N = before.shape[0]
        for count in xrange(N):
            try:
                maes.append(self.MAE(before[count], after_aggregated, t))
                rmses.append(self.RMSE(before[count], after_aggregated, t))
            except Exception:
                pass

        return sum(maes) / N, sum(rmses) / N, max(maes), max(rmses)



if __name__ == "__main__":
    gr = GroupRecommender('test_dataset')
    gr.load_local_data('test_dataset', 100, 0)


    # result = gr.predict_for_group_merging_profiles()
    #
    # result = gr.evaluate(aggregation="average", method='before')
    # print gr.evaluate_aggregation_before()

    # result = gr.predict_for_group_merging_profiles()
    # cProfile.run("result = gr.predict_for_group_merging_recommendations(aggregation='additive', threshold=5, top=10, l=3)")
    result = gr.predict_for_group_merging_recommendations(aggregation='average_without_misery', threshold=10, l=2)
    # for i in result:
    #     print i[0], i[1], "\n"
    print result

