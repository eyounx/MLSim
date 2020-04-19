from datetime import date, timedelta, datetime
import os, sys
from components.models import ObservationRatio, InfectRatio, TouchRatio, DeadRatio, DummyModel, IsolationRatio
import pandas as pd
from components.utils import prepareData, codeDict, flowHubei, flowOutData, get_data_path, get_important_date, get_seed_num, get_core_num, clear_child_process, global_seed, get_populations
from components.simulator import Simulator, InfectionList
from show_simulation_result import run_simulation
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../ncovmodel"))

def run_opt(city, budget, start_date, important_dates, infectratio_range=None,
            dummy_range=None, unob_flow_num=None, repeat_time=1, init_samples=None,training_date_end=None,
            json_name='data_run.json', seed=1, loss_ord=0.0, touch_range=None, iso_range=None):
    if city == 420000:
        infectratio_range = [0., 0.1]
        dummy_range = [0.0000, 400.00001]
    else:
        assert infectratio_range is not None and dummy_range is not None
    days_predict = 0
    # load data
    real_data = pd.read_csv(get_data_path())

    history_real = prepareData(real_data)
    flow_out_data = flowOutData()
    # initialize models

    infectratio = InfectRatio(1, [infectratio_range], [True])
    if touch_range is None:
        touchratio = TouchRatio(1, [[0.0, 0.3333]], [True])
    else:
        touchratio = TouchRatio(1, [touch_range], [True])
    touchratiointra = TouchRatio(1, [[0., 1.]], [True])
    obs = ObservationRatio(1, [[0.0, 0.3]], [True]) #0.2 0.3
    dead = DeadRatio(1, [[0., 0.01]], [True])
    if iso_range is None:
        isoratio = IsolationRatio(1, [[0.03, 0.24]], [True]) # 0.03 0.12  0.24
    else:
        isoratio = IsolationRatio(1, [iso_range], [True]) # 0.03 0.12  0.24

    dummy = DummyModel(1, [dummy_range], [True, True])
    cure_ratio = InfectRatio(1, [[0., 0.1]], [True])


    # set the time of applying touchratio
    simulator = Simulator(city, infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra, cure_ratio, important_dates,
                          unob_flow_num=unob_flow_num, flow_out_data=flow_out_data, training_date_end=training_date_end)
    test_date = datetime.strptime(history_real['date'].max(), '%Y-%m-%d').date() - timedelta(days_predict)
    history_real = history_real[history_real['adcode'] == city]
    history_real = history_real[history_real['date'] >= str(start_date)]
    history_train = history_real[history_real['date'] <= str(test_date)]

    x, y = simulator.fit(history_train, budget=budget, server_num=get_core_num(),
                         repeat=repeat_time, seed=seed, intermediate_freq=10000, init_samples=init_samples, loss_ord=loss_ord)
    print('best_solution: x = ', x, 'y = ', y)
    simulator.set_param(x)
    run_simulation(x, city, 90, 90, start_date, get_important_date(city), unob_flow_num=unob_flow_num, json_name=json_name)
    return x


def initHubei(x, start_date, important_date, travel_from_hubei):
    if travel_from_hubei is None:
        return None
    days_predict = 0
    # load data
    real_data = pd.read_csv(get_data_path())
    history_real = prepareData(real_data)

    infectratio = InfectRatio(1, [[0, 1]], [True])
    touchratio = TouchRatio(1, [[0., 0.3]], [True])
    touchratiointra = TouchRatio(1, [[0, 10]], [True])
    obs = ObservationRatio(1, [[0.0, 1.]], [True])
    dead = DeadRatio(1, [[0., 0.1]], [True])
    isoratio = IsolationRatio(1, [[0., 1]], [True])
    cure_ratio = InfectRatio(1, [[0., 100]], [True])
    dummy = DummyModel(1, [[0, 200000]], [True, True])

    flow_out_data = flowOutData()

    test_date = datetime.strptime(history_real['date'].max(), '%Y-%m-%d').date() - timedelta(days_predict)
    history_real = history_real[history_real['adcode'] == 420000]
    history_real = history_real[history_real['date'] >= str(start_date)]
    history_train = history_real[history_real['date'] <= str(test_date)]
    duration = len(history_train["date"].unique())
    city = 420000

    simulator = Simulator(city, infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra, cure_ratio,
                          important_date, flow_out_data=flow_out_data)
    simulator.set_param(x)
    total_population = get_populations()[420000]
    simulated_result, detailed_result = simulator.simulate(str(start_date), duration+60)
    init_unob = [item for item in reversed(detailed_result.unob_total_init)]
    unob_ratio = {}
    for i, item in enumerate(init_unob):
        date_now = start_date - timedelta(i+1)
        unob_ratio[date_now] = item / total_population
    for it_date, unob_num in zip(simulated_result['date'], simulated_result['total_unobserved']):
        unob_ratio[datetime.strptime(it_date, '%Y-%m-%d').date()] = unob_num/total_population
    unob_flow_num = {}
    lst_travel = {}
    lst_date = None
    for it_date in travel_from_hubei:
        if it_date not in unob_ratio:
            continue
        if it_date not in unob_flow_num:
            unob_flow_num[it_date] = {}
        for it_code in travel_from_hubei[it_date]:
            unob_flow_num[it_date][it_code] = travel_from_hubei[it_date][it_code] * unob_ratio[it_date]
        lst_travel = travel_from_hubei[it_date]
        lst_date = it_date
    lst_date = lst_date + timedelta(1)
    while True:
        if lst_date in unob_ratio:
            if lst_date not in unob_flow_num:
                unob_flow_num[lst_date] = {}
            for it_code in lst_travel:
                unob_flow_num[lst_date][it_code] = lst_travel[it_code] * unob_ratio[lst_date]
        else:
            break
        lst_date = lst_date + timedelta(1)
    return unob_flow_num


if __name__ == '__main__':
    province_travel_dict = flowHubei()
    real_data = pd.read_csv(get_data_path())['adcode'].unique()
    province_code_dict = codeDict()
    training_end_date = None
    x_buffer = []

    rerun_list = None
    if rerun_list is not None:
        old_param_buffer = [pd.read_csv('params{}.csv'.format(int(i))) for i in range(get_seed_num())]
    for i in range(get_seed_num()):
        global_seed(i)
        all_param = {}
        if rerun_list is None or 420000 in rerun_list:
            x = run_opt(420000, 200000, start_date=date(2020, 1, 11), important_dates=[get_important_date(420000)],
                repeat_time=1, training_date_end=training_end_date, seed=i, json_name='data_run{}.json'.format(int(i)))
            clear_child_process()
            all_param[420000] = x
            x_buffer.append(x)
        else:
            x = list(old_param_buffer[i][str(420000)])
            x_buffer.append(x)
            all_param[420000] = x

        unob_flow_num = initHubei(x_buffer[i], start_date=date(2020, 1, 11), important_date=[get_important_date(420000)],
                                  travel_from_hubei=province_travel_dict)
        for ind, item in enumerate(real_data):
            print(i, ind, item, province_code_dict[item])
            if item == 420000:
                continue
            if rerun_list is None or item in rerun_list:
                x = run_opt(item, 40000, start_date=date(2020, 1, 11), important_dates=[get_important_date(item)],
                        infectratio_range=[0., 0.05], dummy_range=[0, 0.000001], unob_flow_num=unob_flow_num, repeat_time=1,
                        training_date_end=training_end_date, seed=i, json_name='data_run{}.json'.format(int(i)),
                            iso_range=[0.03, 0.12])
                clear_child_process()
                all_param[item] = x
            else:
                all_param[item] = list(old_param_buffer[i][str(item)])
        all_param_df = pd.DataFrame(all_param)
        all_param_df.to_csv('params{}.csv'.format(int(i)), index=False)

