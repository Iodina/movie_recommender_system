import numpy as np
from rec_sys import Recommender


class Aggregation(Recommender):
    def __init__(self, datafile_path=None):
        Recommender.__init__(self, datafile_path=datafile_path)
        self.predictions = np.array([1,3,4,5],[4,2,1,2])
        self.aggregation_function = {'additive': self.additive, 'multiplicative': self.multiplicative}

    def predict_for_group(self, aggregation='additive', parameters=None):
        # self.predictions = Recommender.predict_for_all_fake_users(self)
        self.predictions = [
            {'i1': 10, 'i2': 4, 'i3': 3, 'i4': 6, 'i5': 10,
             'i6': 9, 'i7': 6, 'i8': 8, 'i9': 10, 'i10': 8},
            {'i1': 1, 'i2': 9, 'i3': 8, 'i4': 9, 'i5': 7,
             'i6': 9, 'i7': 6, 'i8': 9, 'i9': 3, 'i10': 8},
            {'i1': 10, 'i2': 5, 'i3': 2, 'i4': 7, 'i5': 9,
             'i6': 8, 'i7': 5, 'i8': 6, 'i9': 7, 'i10': 6}]
        agregate = self.aggregation_function.get(aggregation)
        return agregate(self.predictions, parameters)

    def additive(self, parameters=None):
        sum_array = self.predictions.sum(axis=0)
        return np.multiply(sum_array, 1/np.linalg.norm(sum_array))

    def multiplicative(self, parameters=None):
        prod_array = self.predictions.prod(axis=0)
        return np.multiply(prod_array, 1/np.linalg.norm(prod_array))

    def

