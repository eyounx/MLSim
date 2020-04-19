import torch
from torch import nn
import pandas as pd
import os
from datetime import date, datetime
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib
font = {'size'   : 18}
matplotlib.rc('font', **font)
from    datetime import timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from run import run_opt as run_opt_china, initHubei
from components.utils import codeDict, flowHubei, get_data_path, get_important_date, get_seed_num
from run_foreign import run_opt as run_opt_foreign
from show_simulation_foreign import run_simulation as run_simulation_foreign
from res_analysis import load_and_save
import argparse

korea_flag = False
foresee_size = 3

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load_inter', type=bool, default=False)
    args = parser.parse_args()
    return args


if korea_flag:
    start_date = date(2020, 1, 26)
    day_num = []
    date_it = date(2020, 3, 5)
    period_it = (date_it - start_date).days
    training_end_date = date_it
else:
    date_it = date(2020, 2, 13)
    start_date = date(2020, 1, 14)
    period_it = (date_it - start_date).days
    training_end_date = date_it


def ChinaSEIRData(city=420000):
    d = {
    }
    if city in d:
        return d[city]
    else:
        data_all = json.load(open('./data/data-seir.json', 'r'))
        if str(city) in data_all:
            return data_all[str(city)]
        return [0] * len(data_all[str(420000)])


def HanSEIRData():
    data_all = json.load(open('./data/data-seir-korea.json', 'r'))
    return data_all['900003']

def plot1(ours, real, simulated, seir, title, savefile, date_it=None, loc='upper left', y_max2=None, interval=12):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    y_max = -1000
    y_max = max(np.max(ours), np.max(real), np.max(simulated), np.max(seir))

    xs = [start_date + timedelta(days=i) for i in range(len(real))]
    plt.plot(xs, real, color="indianred", linewidth=2.5, linestyle="-", label="real")
    xs = [start_date + timedelta(days=i) for i in range(len(simulated))]
    plt.plot(xs,simulated, color="orange", linewidth=2.5, linestyle="-",  label="LSTM")
    xs = [start_date + timedelta(days=i) for i in range(len(ours))]
    plt.plot(xs,ours, color="cornflowerblue", linewidth=2.5, linestyle="-",  label="MLSim")
    xs = [start_date + timedelta(days=i) for i in range(len(seir))]
    plt.plot(xs,seir, color="forestgreen", linewidth=2.5, linestyle="-",  label="SEIR")
    if date_it is not None:
        plt.vlines(date_it, 0, y_max, colors="r", linestyles="dashed")
    plt.gcf().autofmt_xdate()
    #plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Cases number', fontsize=18)
    ax=plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    plt.ylim(bottom=0)
    if y_max2 is not None:
        plt.ylim(top=y_max2)
    #ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(loc=loc)
    #plt.title(title)
    plt.savefig(savefile, bbox_inches='tight')
    plt.clf()

def get_total(data, type, cities):
    l = len(data[type][str(cities[0])])
    res = l * [0]
    for ind, city in enumerate(cities):
        for i in range(l):
            res[i] += data[type][str(city)][i]
    return res

def get_COVID_new_confirm(type='real_confirmed', cities=420000, file='./data_run_lstm_china0.json'):
    with open(file, 'r') as f:
        data = json.load(f)
    return np.array(get_total(data, type, [int(cities)]))

def prepare_data():
    #smooth_confirm = json.load(open('./data/sars.json', 'r'))['data']
    smooth_confirm = [0] * 116
    smooth_confirm = np.array([np.around(item) for item in smooth_confirm])
    return smooth_confirm

def train_china():
    province_travel_dict = flowHubei()
    real_data = pd.read_csv(get_data_path())['adcode'].unique()
    province_code_dict = codeDict()
    for i in range(1):
        all_param = {}
        x = run_opt_china(420000, 200000, start_date=date(2020, 1, 11), important_dates=[get_important_date(420000)],
                          repeat_time=3, training_date_end=training_end_date, seed=i,
                          json_name='data_run_lstm_china{}.json'.format(int(i)), loss_ord=4., touch_range=[0, 0.33])
        unob_flow_num = initHubei(x, start_date=date(2020, 1, 11), important_date=[get_important_date(420000)],
                                  travel_from_hubei=province_travel_dict)
        all_param[420000] = x
        real_data = [110000, 440000, 330000, 310000, 320000, 120000]
        for ind, item in enumerate(real_data):
            print(i, ind, item, province_code_dict[item])
            if item == 420000:
                continue
            x = run_opt_china(item, 40000, start_date=date(2020, 1, 11), important_dates=[get_important_date(420000)],
                        infectratio_range=[0.0, 0.05], dummy_range=[0, 0.000001], unob_flow_num=unob_flow_num, repeat_time=2,
                        training_date_end=training_end_date, json_name='data_run_lstm_china{}.json'.format(int(i)),
                              loss_ord=4., touch_range=[0.0, 0.33], iso_range=[0.03, 0.12])
            all_param[item] = x
        all_param_df = pd.DataFrame(all_param)
        all_param_df.to_csv('params_lstm_china{}.csv'.format(int(i)), index=False)
    load_and_save('data_run_lstm_china{}.json', 'data_run_lstm_china_{}.json', 1)

def train_korea():
    province_code_dict = {900003: 'South Korea'}
    for i in range(1):
        all_param = {}
        for ind, item in enumerate([900003]):
            print(i, ind, item, province_code_dict[item])
            x = run_opt_foreign(item, 40000, start_date=date(2020, 1, 24), important_dates=[get_important_date(900003)],
                        infectratio_range=[0.0, 0.05], dummy_range=[0, 100], unob_flow_num=None, repeat_time=3,
                        training_end_date=training_end_date, seed=i, json_name='data_run_lstm_korea{}.json'.format(int(i)),
                                touch_range=[0, 0.333])
            run_simulation_foreign(x, item, 60, 60,
                                   start_date, get_important_date(item), json_name='data_run_lstm_korea{}.json'.format(int(i)))
            all_param[item] = x
        all_param_df = pd.DataFrame(all_param)
        all_param_df.to_csv('params_lstm_korea{}.csv'.format(int(i)), index=False)
    load_and_save('data_run_lstm_korea{}.json', 'data_run_lstm_korea_{}.json', 1)


def construct_data_set(data, size=3):
    dataX = []
    dataY = []
    for i in range(len(data) - size):
        x_it = data[i:i+size]
        y_it = data[i+size]
        dataX.append(x_it)
        dataY.append(y_it)
    dataX = np.array(dataX)
    dataY = np.array(dataY).reshape((-1, 1))
    return dataX, dataY


class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(lstm_reg, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, mem_in=None):
        x = torch.tanh(self.fc1.forward(x))
        x, mem = self.rnn(x,mem_in)
        x = torch.tanh(self.fc2(x))
        return x, mem

    def forward_and_pred(self, x, pred=0):
        output, mem = self.forward(x)
        if pred > 0:
            current_x = list(x[0][-1].data.numpy())
            current_y = output[0][-1].data.numpy()[0]
            res = []
            for i in range(pred):
                current_x.pop(0)
                current_x.append(current_y)
                net_x = torch.from_numpy(np.array(current_x).reshape((1,1,-1)))
                ot, mem = self.forward(net_x, mem)
                res.append(ot)
                current_y = ot[0][-1].data.numpy()[0]
            out_new = torch.cat(res, -2)
            output = torch.cat([output, out_new], -2)
        return output


def train_and_pred(trainX, trainY, testX, pred_num=30,den=4500,load_model=False,mask=None):
    net = lstm_reg(foresee_size, 30, 1, 1)
    if load_model:
        net.load_state_dict(torch.load('params.pkl'))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    train_x, train_y = torch.from_numpy(trainX).to(torch.float32), torch.from_numpy(trainY).to(
        torch.float32)
    test_x = torch.from_numpy(testX).to(torch.float32).unsqueeze(0)
    train_x = train_x / den
    train_y = train_y / den
    test_x = test_x / den
    if mask is not None:
        mask_tensor =torch.from_numpy(mask).to(torch.float32).detach()
    if not load_model:
        for i in range(500):
            pred, _ = net.forward(train_x)
            if mask is None:
                loss = criterion(pred, train_y)
            else:
                d1 = list(np.round((torch.sqrt((pred - train_y)**2) * mask_tensor).squeeze()[1].detach().numpy(),3))
                d2 = list(np.round((pred * mask_tensor).squeeze()[1].detach().numpy(), 3))
                d3 = list(np.round((train_y * mask_tensor).squeeze()[1].detach().numpy(), 3))
                pred_test = list(np.round(net.forward_and_pred(test_x, 0).squeeze().data.numpy(), 3))
                loss = (((pred - train_y)**2) * mask_tensor).sum() / mask_tensor.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(i, loss.item())
        #torch.save(net.state_dict(), 'params.pkl')
    pred_train = net.forward_and_pred(train_x, 0).data.numpy()
    pred_test = net.forward_and_pred(test_x, pred_num).data.numpy()
    return pred_train, pred_test[0]



def main(load_inter_flag=False):
    if not load_inter_flag:
        if korea_flag:
            train_korea()
        else:
            train_china()
    fmt = 'pdf'
    #all_city = pd.read_csv(get_data_path())['adcode'].unique()
    all_city = [420000, 110000, 440000, 330000, 310000, 320000, 120000]
    if korea_flag:
        all_city = [900003]
    mlsim_loss_array = []
    seir_loss_array = []
    lstm_loss_array = []
    for city in all_city:
        print(city)
        torch.manual_seed(6)
        if not os.path.exists('./img/{}{}/'.format('./lstm', str(city))):
            os.mkdir('./img/{}{}/'.format('./lstm', str(city)))
        if not korea_flag:
            div_idx = period_it
            dataX_list, dataY_list = [], []
            mask_list = []
            _DataX, _DataY = construct_data_set(prepare_data(), foresee_size)
            _DataX, _DataY = np.expand_dims(_DataX, 0), np.expand_dims(_DataY, 0)
            dataX_list.append(_DataX)
            dataY_list.append(_DataY)
            mask_list.append(np.zeros((1, np.shape(_DataY)[1], 1)))
            TestDataX, TestDataY =construct_data_set(get_COVID_new_confirm(cities=city), foresee_size)
            mask_list.append(np.zeros((1, np.shape(_DataY)[1], 1)))
            dataX_list.append(np.zeros((1, np.shape(_DataY)[1], foresee_size)))
            dataY_list.append(np.zeros((1, np.shape(_DataY)[1], 1)))
            for i in range(div_idx):
                mask_list[-1][0][i][0] = 1
                dataY_list[-1][0][i][0] = TestDataY[i][0]
                for ii in range(foresee_size):
                    dataX_list[-1][0][i][ii] = TestDataX[i][ii]
            DataX = np.concatenate(dataX_list, 0)
            DataY = np.concatenate(dataY_list, 0)
            mask = np.concatenate(mask_list, 0)
            TestDataX, TestDataY =construct_data_set(get_COVID_new_confirm(cities=city), foresee_size)
            _, real_cum =construct_data_set(get_COVID_new_confirm('real_cum_confirmed',cities=city), foresee_size)
            _, sim_cum =construct_data_set(get_COVID_new_confirm('sim_cum_confirmed_deduction_s1',cities=city), foresee_size)
            _, sim_inc =construct_data_set(get_COVID_new_confirm('sim_confirmed_deduction_s1',cities=city), foresee_size)
            load_model = False
            _, sim_inc_seir = construct_data_set(ChinaSEIRData(city=city)[:-3], foresee_size)
        else:
            div_idx = period_it
            dataX_list, dataY_list = [], []
            mask_list = []
            for item in pd.read_csv(get_data_path())['adcode'].unique():
                _DataX, _DataY =construct_data_set(get_COVID_new_confirm(cities=int(item)), foresee_size)
                _DataX, _DataY =np.expand_dims(_DataX, 0), np.expand_dims(_DataY, 0)
                dataX_list.append(_DataX)
                dataY_list.append(_DataY)
                mask_list.append(np.ones((1,np.shape(_DataY)[1], 1)))
            TestDataX, TestDataY =construct_data_set(get_COVID_new_confirm(file='data_run_lstm_korea0.json', cities=900003), foresee_size)
            mask_list.append(np.zeros((1, np.shape(_DataY)[1], 1)))
            dataX_list.append(np.zeros((1, np.shape(_DataY)[1], foresee_size)))
            dataY_list.append(np.zeros((1, np.shape(_DataY)[1], 1)))
            for i in range(div_idx):
                mask_list[-1][0][i][0] = 1
                dataY_list[-1][0][i][0] = TestDataY[i][0]
                for ii in range(foresee_size):
                    dataX_list[-1][0][i][ii] = TestDataX[i][ii]
            DataX = np.concatenate(dataX_list, 0)
            DataY = np.concatenate(dataY_list, 0)
            mask =np.concatenate(mask_list, 0)
            TestDataX, TestDataY =construct_data_set(get_COVID_new_confirm(file='data_run_lstm_korea0.json', cities=900003), foresee_size)
            _, real_cum =construct_data_set(get_COVID_new_confirm(type='real_cum_confirmed', file='data_run_lstm_korea0.json', cities=900003), foresee_size)
            _, sim_cum =construct_data_set(get_COVID_new_confirm('sim_cum_confirmed_deduction_s1',file='data_run_lstm_korea0.json', cities=900003), foresee_size)
            _, sim_inc =construct_data_set(get_COVID_new_confirm('sim_confirmed_deduction_s1',file='data_run_lstm_korea0.json', cities=900003), foresee_size)
            _, sim_inc_seir = construct_data_set(HanSEIRData()[:], foresee_size)
            load_model = False

        den = 4500
        if city < 900000 and not city == 420000:
            den = 200
        # print(TestDataX)
        if korea_flag:
            pred_train, pred_test = train_and_pred(DataX, DataY, TestDataX[:div_idx][:], len(sim_cum) - div_idx, den, load_model=load_model, mask=mask)
        else:
            pred_train, pred_test = train_and_pred(DataX, DataY, TestDataX[:div_idx][:], len(sim_cum) - div_idx, den, load_model=load_model, mask=mask)
        pred_test = pred_test * den
        pred_test[pred_test < 0] = 0
        if not os.path.exists('./img/lstm'):
            os.mkdir('./img/lstm')
        max_len = len(TestDataY)
        plot1(sim_inc[:max_len], TestDataY, pred_test[:max_len], sim_inc_seir[:max_len],
              'The number of newly confirmed cases ',
              './img/{}{}/newly_real_sim.{}'.format('./lstm',str(city), fmt),
              date_it=date_it,
              loc='upper right',
              )
        plot1(sim_cum, real_cum, np.cumsum(pred_test), np.cumsum(sim_inc_seir),
              'The cumulative number of newly confirmed cases ',
              'img/{}{}/cum_real_sim.{}'.format('./lstm',str(city), fmt),
              date_it=date_it,
              loc='lower right')
        if city == 420000:
            plot1(sim_inc, TestDataY, pred_test, sim_inc_seir,
                  'The number of newly confirmed cases ',
                  './img/{}{}/newly_real_sim.{}'.format('./lstm', str(city), fmt),
                  date_it=date_it,
                  loc='upper right',
                  #y_max2=7000
                  )
        if city == 900003:
            plot1(sim_inc[:max_len], TestDataY, pred_test[:max_len], sim_inc_seir[:max_len],
                  'The number of newly confirmed cases ',
                  './img/{}{}/newly_real_sim.{}'.format('./lstm', str(city), fmt),
                  date_it=date_it,
                  loc='upper left',
                  interval=12
                  )
        train_size = (date_it - start_date).days
        print(train_size, len(TestDataY))
        test_label = np.array([TestDataY[item] for item in range(train_size)])
        loss_ours = np.sqrt(np.mean(np.square(test_label - np.array([sim_inc[item] for item in range(train_size)]))))
        loss_seir = np.sqrt(
            np.mean(np.square(test_label - np.array([sim_inc_seir[item] for item in range(train_size)]))))
        loss_lstm = np.sqrt(np.mean(np.square(test_label - np.array([pred_test[item] for item in range(train_size)]))))
        print('training, ',city, ' ours: ', loss_ours,
              'seir: ', loss_seir,
              'lstm: ', loss_lstm)
        mlsim_loss_array.append(loss_ours)
        seir_loss_array.append(loss_seir)
        lstm_loss_array.append(loss_lstm)
        test_label =np.array([TestDataY[item] for item in range(train_size, len(TestDataY))])
        loss_ours = np.sqrt(np.mean(np.square(test_label-np.array([sim_inc[item] for item in range(train_size, len(TestDataY))]))))
        loss_seir = np.sqrt(np.mean(np.square(test_label-np.array([sim_inc_seir[item] for item in range(train_size, len(TestDataY))]))))
        loss_lstm = np.sqrt(np.mean(np.square(test_label-np.array([pred_test[item] for item in range(train_size, len(TestDataY))]))))
        print('testing, ours: ', loss_ours,
              'seir: ', loss_seir,
              'lstm: ', loss_lstm)

        mlsim_loss_array.append(loss_ours)
        seir_loss_array.append(loss_seir)
        lstm_loss_array.append(loss_lstm)

    print('MLSim  ',sep='', end='')
    for lsim in mlsim_loss_array:
        print('& ${:.2f}$  '.format(lsim), sep='', end='')
    print('\\\\')
    print('SEIR  ', sep='', end='')
    for lsim in seir_loss_array:
        print('& ${:.2f}$  '.format(lsim), sep='', end='')
    print('\\\\')
    print('LSTM  ', sep='', end='')
    for lsim in lstm_loss_array:
        print('& ${:.2f}$  '.format(lsim), sep='', end='')
    print('\\\\')

    pass

if __name__ == '__main__':
    args = parse_args()
    main(args.load_inter)
