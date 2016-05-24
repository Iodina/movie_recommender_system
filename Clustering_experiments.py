from rec_sys import Recommender
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from scipy.stats import pearsonr
import numpy as np

if __name__ == "__main__":
    recommender = Recommender()
    recommender.load_local_data('group_dataset', K=100, min_values=0)
    raiting_matrix = recommender.matrix.get_rating_matrix()

    from guppy import hpy
    h = hpy()
    print h.heap()

    predictred_raiting_matrix = recommender.get_predictions_for_all_users()
    # np.savetxt("prediction.txt", m1)


    print recommender.matrix.indexes_with_fake_user_ids.keys() # 109 - fake user id

    for n in range(2, 10):
        print "Experiment: k-means, %s number of clusters "%(n, )
        k_means = KMeans(n_clusters=n)
        k_means.fit(predictred_raiting_matrix)
        for i in range(n):
            print "sum of elements: %s"%(np.sum(k_means.labels_ == i),) +  "cluster# %s"%(i,)
        print len(k_means.labels_), k_means.labels_
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print '____________________________________________________'


    # print "centers: %s" %(k_means.cluster_centers_,)
    # np.savetxt("result1", k_means.cluster_centers_)
    # print "labels: %s" %(k_means..labels_,)
    # print "inertia: %s" %(k_means..inertia_,)



    mean_shift = MeanShift(cluster_all=False)
    mean_shift.fit(predictred_raiting_matrix)
    labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_

    for i in range(-1, len(np.unique(labels))):
        print "sum of elements: %s"%(np.sum(labels == i),) +  "  cluster# %s"%(i,)
    print labels
    print("Number of estimated clusters:", len(np.unique(labels)))
    print '____________________________________________________'



    def pearson_affinity(M):
        return 1 - np.array([[pearsonr(a, b)[0] for a in M] for b in M])

    for n in range(2, 10):
        print "Experiment: AgglomerativeClustering, %s number of clusters "%(n, )
        cluster = AgglomerativeClustering(n_clusters=n, linkage='average', affinity=pearson_affinity)
        cluster.fit(predictred_raiting_matrix)
        for i in range(n):
            print "sum of elements: %s"%(np.sum(cluster.labels_ == i),) +  "  cluster# %s"%(i,)
        print cluster.labels_
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print '____________________________________________________'
