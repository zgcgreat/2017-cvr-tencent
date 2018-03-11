import subprocess
from datetime import datetime


def xgb_train(train, test, model_in, model_out):
    cmd = '~/ffm/lightgbm config=train.conf data={train} valid_data={test} input_model={model_in} output_model={model_out}'\
        .format(train=train, test=test, model_in=model_in, model_out=model_out)
    subprocess.call(cmd, shell=True)


def xgb_predict(test, model_in):
    cmd = '~/ffm/lightgbm config=predict.conf input_model={model_in}'.format(test=test, model_in=model_in)
    subprocess.call(cmd, shell=True)


def train(start_date, end_date, num_round):
    for date in range(start_date, end_date+1):
        print('train:{0}, eval:{1}\n'.format(date, date + 1))

        train = in_path+'train_{0}.csv'.format(date)
        eval = in_path + 'train_{0}.csv'.format(date+1)
        model_in = out_path + 'model_{0}.model'.format(date-1)
        model_out = out_path + 'model_{0}.model'.format(date)
        if date == start_date and num_round == 1:
            model_in = ""
        # if date == start_date and num_round > 1:
        #     model_in = 'model_{0}.model'.format(end_date)
        xgb_train(train, eval, model_in, model_out)


def submission():
    with open('../../output/lgb/submission.csv', 'w') as fo:
        fo.write('instanceID,prob\n')
        fi = open('../../output/lgb/lgb_pred.txt', 'r')
        for t, row in enumerate(fi, start=1):
            fo.write(str(t)+','+str(row.replace('\n', '')) + '\n')

if __name__ == '__main__':
    start = datetime.now()
    in_path = '../../data/daily_data/'
    out_path = '../../output/lgb/model/'
    start_date = 27
    end_date = 28

    for round in range(1, 3):
        train(start_date, end_date, round)

    test = in_path + 'test.csv'
    pred_model = out_path + 'model_{0}.model'.format(end_date)
    xgb_predict(test, pred_model)

    submission()

    print('耗时:', datetime.now()-start)

