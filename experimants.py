import xlwt
from social_choice import GroupRecommender

# TODO Create L
# TODO modify before
# TODO float


def save_to_xls(gr=None, title='experiment'):
    style0 = xlwt.easyxf('font: name Times New Roman, color-index green, bold on')
    wb = xlwt.Workbook()


    ws = wb.add_sheet('Evaluation Experiment')
    ws.write(0, 0, 'Method', style0)
    ws.write(1, 0, 'Merging Recommendations', style0)



    ws.write(0, 1, 'Ev average', style0)
    ws.write(0, 2, 'Ev misery', style0)

    ws.write(0, 8, '%s' % (len(gr.matrix.indexes_with_fake_user_ids.keys()),))
    ws.write(0, 9, '%sx%s' % (gr.matrix.rating_matrix.shape[0], gr.matrix.rating_matrix.shape[1]))
# 'copeland',

    # ranking metrics

    list_L = ['fairness', 'plurality_voting']
    list_T = ['average_without_misery','approval_voting']
    list_all = ['additive', 'multiplicative', 'average', 'borda', 'least_misery', 'most_pleasure']

    l = gr.matrix.rating_matrix.shape[1]
    threshold = 1

    i = 2
    for fun in list_all:
        res = gr.evaluate(aggregation=fun)
        ws.write(i, 0, '%s'%(fun,))
        ws.write(i, 1, '%s'%(res[0],))
        ws.write(i, 2, '%s'%(res[1],))
        i += 1

    i += 1
    for fun in list_T:
        ws.write(i, 0, '%s'%(fun,), style0)
        ws.write(i+1, 0, 'T')
        for k in range(10):

            ws.write(i+1, k+1, '%s'%(threshold,))
            res = gr.evaluate(aggregation=fun, threshold=threshold)
            ws.write(i+2, k+1, '%s'%(res[0],))
            ws.write(i+3, k+1, '%s'%(res[1],))
            threshold += 1
        i += 4
        threshold = 1

    i += 1
    for fun in list_L:
        ws.write(i, 0, '%s'%(fun,), style0)
        ws.write(i+1, 0, 'L')
        for k in range(10):
            l = int(l/100 + k*100)
            ws.write(i+1, k+1, '%s'%(l,))
            res = gr.evaluate(aggregation=fun, l=l)
            ws.write(i+2, k+1, '%s'%(res[0],))
            ws.write(i+3, k+1, '%s'%(res[1],))
        i += 4
        l = gr.matrix.rating_matrix.shape[1]

    i += 1


    res = gr.evaluate_aggregation_after('average')

    ws.write(i, 0, 'Accuracy', style0)
    ws.write(i+1, 0, 'After, average', style0)
    ws.write(i+2, 0, 'MAE')
    ws.write(i+3, 0, 'RMSE')
    ws.write(i+2, 1, res[0])
    ws.write(i+3, 1, res[1])
    ws.write(i+2, 2, res[2])
    ws.write(i+3, 2, res[3])

    i += 5

    ws.write(i, 0, 'After av_without', style0)
    ws.write(i+1, 0, 'T')
    ws.write(i+2, 0, 'MAE')
    ws.write(i+4, 0, 'RMSE')

    for k in range(10):
        ws.write(i+1, k+1, '%s'%(threshold,))
        res = gr.evaluate_aggregation_after('average', threshold)
        ws.write(i+2, k+1, res[0])
        ws.write(i+3, k+1, res[2])
        ws.write(i+4, k+1, res[1])
        ws.write(i+5, k+1, res[3])
        threshold += 1


    threshold = 1

    i += 11

    #this changes matrix

    res = gr.evaluate(method='before')

    ws.write(i, 0, 'Before, raking', style0)
    ws.write(i+1, 1, res[0])
    ws.write(i+1, 2, res[1])

    i += 3

    res = gr.evaluate_aggregation_before()

    ws.write(i, 0, 'Before, accuracy', style0)
    ws.write(i+1, 0, 'MAE')
    ws.write(i+2, 0, 'RMSE')
    ws.write(i+1, 1, res[0])
    ws.write(i+2, 1, res[1])
    ws.write(i+1, 2, res[2])
    ws.write(i+2, 2, res[3])


    wb.save('%s.xls' % (title,))


if __name__ == "__main__":
    # gr = GroupRecommender('test_dataset')
    # save_to_xls(gr=gr, title='exp/test')


    gr = GroupRecommender('2_users_dataset_3')
    save_to_xls(gr=gr, title='exp/experiment_3_filtered')

    gr = GroupRecommender('2_users_6fake_dataset_3')
    save_to_xls(gr=gr, title='exp/experiment_6fake_filtered')

    gr = GroupRecommender('2_users_9fake_dataset_3')
    save_to_xls(gr=gr, title='exp/experiment_9fake_filtered')

    # gr = GroupRecommender('5_users_dataset_7')
    # save_to_xls(gr=gr, title='exp/experiment_5_filtered')
    #
    # gr = GroupRecommender('8_users_dataset_15')
    # save_to_xls(gr=gr, title='exp/experiment_8_filtered')



