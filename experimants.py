import xlwt
from social_choice import GroupRecommender


def save_to_xls(gr=None, title='experiment'):
    style0 = xlwt.easyxf('font: name Times New Roman, color-index green, bold on')
    wb = xlwt.Workbook()
    l = gr.matrix.rating_matrix.shape[1]
    threshold = 4.6
    for n in range(5):
        l = int(l/100. + n*50)
        threshold += 1
        ws = wb.add_sheet('Evaluation Experiment %s'%(n,))
        ws.write(0, 0, 'Method', style0)
        ws.write(1, 0, 'Merging Recommendations', style0)

        ws.write(13, 0, 'Merging Profiles', style0)
        ws.write(14, 0, 'average')

        ws.write(0, 1, 'Parameter', style0)
        # ws.write(5, 1, 'Threshold=%s' % (threshold,))
        # ws.write(8, 1, 'L=%s' % (l,))
        # ws.write(11, 1, 'L=%s' % (l,))
        # ws.write(12, 1, 'Threshold=%s' % (threshold,))

        ws.write(0, 2, 'Group Size', style0)

        group_size = len(gr.matrix.indexes_with_fake_user_ids.keys())
        for i in range(2, 13):
            ws.write(i, 2,
                     '%s' % (group_size,))

        ws.write(14, 2,
                     '%s' % (group_size,))

        ws.write(0, 3, 'Matrix', style0)
        for i in range(2, 13):
            ws.write(i, 3, '%sx%s' % (
            gr.matrix.rating_matrix.shape[0], gr.matrix.rating_matrix.shape[1]))

        ws.write(14, 3, '%sx%s' % (
            gr.matrix.rating_matrix.shape[0], gr.matrix.rating_matrix.shape[1]))

        ws.write(0, 4, 'Ev average', style0)
        ws.write(0, 5, 'Ev misery', style0)

        for i in range(11):
            aggr_fun_name = gr.aggregation_function.items()[i][0]
            if not aggr_fun_name == 'copeland':
                res = gr.evaluate(aggregation=aggr_fun_name, l=l,
                            threshold=threshold)
                ws.write(i + 2, 0, '%s' % (aggr_fun_name,))
                ws.write(i + 2, 4, '%s' % (res[0],))
                ws.write(i + 2, 5, '%s' % (res[1],))
            if aggr_fun_name in ['fairness', 'plurality_voting']:
                ws.write(i+2, 1, 'L=%s' % (l,))
            if aggr_fun_name in ['average_without_misery', 'approval_voting']:
                ws.write(i+2, 1, 'Threshold=%s' % (threshold,))

    eval = gr.evaluate(method='before')
    ws.write(14, 4, '%s'%(eval[0],))
    ws.write(14, 5, '%s'%(eval[0],))

    wb.save('%s.xls' % (title,))


if __name__ == "__main__":
    gr = GroupRecommender('2_users_dataset')
    save_to_xls(gr=gr, title='exp/experiment_3_full')

    gr = GroupRecommender('2_users_dataset_3')
    save_to_xls(gr=gr, title='exp/experiment_3_filtered')

    gr = GroupRecommender('5_users_dataset_7')
    save_to_xls(gr=gr, title='exp/experiment_5_filtered')

    gr = GroupRecommender('8_users_dataset_15')
    save_to_xls(gr=gr, title='exp/experiment_8_filtered')



