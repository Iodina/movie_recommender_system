import numpy as np
from rec_sys import Recommender

        # self.predictions = [
        #     {'i1': 10, 'i2': 4, 'i3': 3, 'i4': 6, 'i5': 10,
        #      'i6': 9, 'i7': 6, 'i8': 8, 'i9': 10, 'i10': 8},
        #     {'i1': 1, 'i2': 9, 'i3': 8, 'i4': 9, 'i5': 7,
        #      'i6': 9, 'i7': 6, 'i8': 9, 'i9': 3, 'i10': 8},
        #     {'i1': 10, 'i2': 5, 'i3': 2, 'i4': 7, 'i5': 9,
        #      'i6': 8, 'i7': 5, 'i8': 6, 'i9': 7, 'i10': 6}]

class GroupRecommender(Recommender):
    def __init__(self, datafile_path=None):
        Recommender.__init__(self, datafile_path=datafile_path)
        self.predictions = np.array([1,3,4,5],[4,2,1,2])
        self.aggregation_function = {'additive': self.additive, 'multiplicative': self.multiplicative}

    def _norm(self, vector):
        return np.linalg.norm(vector)

    def predict_for_group(self, aggregation='additive', parameters=None):
        self.predictions = self.predicted_rating_submatrix_for_fake()
        aggregate = self.aggregation_function.get(aggregation)
        return aggregate(self.predictions, parameters)

    def additive(self, parameters=None):
        sum_array = self.predictions.sum(axis=0)
        return np.multiply(sum_array, 1/self._norm(sum_array))

    def multiplicative(self, parameters=None):
        prod_array = self.predictions.prod(axis=0)
        return np.multiply(prod_array, 1/self._norm(prod_array))

    def average(self, parameters=None):
        sum_array = self.predictions.sum(axis=0)
        length = len(sum_array)
        return np.multiply(sum_array, 1./length)

    def average_without_misery(self, parameters={'threshold': 3}):
        predict = self.predictions
        threshold = parameters.get('threshold')
        for item in self.predictions.swapaxes(0,1):
            list_of_indx = [i for i in item if i < threshold]
            np.delete(predict, obj=list_of_indx, axis=1)
        sum_array = predict.sum(axis=0)

        return np.multiply(sum_array, 1/self._norm(sum_array)


if __name__ == "__main__":
    gr = GroupRecommender('dataset')
    gr.load_local_data('dataset', 100, 0)
