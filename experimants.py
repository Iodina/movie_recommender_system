import xlwt
from social_choice import GroupRecommender


def save_to_xls(gr=GroupRecommender(), title='experiment', l=3, threshold=5):
    style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on')
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Evaluation Experiment')
    ws.write(0, 0, 'Method', style0)
    ws.write(1, 0, 'Merging Recommendations', style0)

    ws.write(13, 0, 'Merging Profiles', style0)
    ws.write(14, 0, 'average')

    ws.write(0, 1, 'Parameter', style0)
    ws.write(5, 1, 'Threshold=%s' % (threshold,))
    ws.write(8, 1, 'L=%s' % (l,))
    ws.write(11, 1, 'L=%s' % (l,))
    ws.write(12, 1, 'Threshold=%s' % (threshold,))

    ws.write(0, 2, 'Group Size', style0)
    for i in range(2, 13):
        ws.write(i, 2,
                 '%s' % (len(gr.matrix.indexes_with_fake_user_ids.keys()),))

    ws.write(14, 2,
                 '%s' % (len(gr.matrix.indexes_with_fake_user_ids.keys()),))

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
        ws.write(i + 2, 0, '%s' % (aggr_fun_name,))
        ws.write(i + 2, 4, '%s' % (
        gr.evaluate(aggregation=aggr_fun_name, l=l,
                    threshold=threshold)[0],))
        ws.write(i + 2, 5, '%s' % (
        gr.evaluate(aggregation=aggr_fun_name, l=l,
                    threshold=threshold)[1],))

    eval = gr.evaluate(method='before')
    ws.write(14, 4, '%s'%(eval[0],))
    ws.write(14, 5, '%s'%(eval[0],))

    wb.save('%s.xls' % (title,))


if __name__ == "__main__":
    gr = GroupRecommender('test_dataset')
    gr.load_local_data('test_dataset', 100, 0)
    save_to_xls(gr=gr, title='exp/experiment_full', l=3, threshold=5)
