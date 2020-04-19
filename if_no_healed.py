from datetime import date, timedelta, datetime
import os, sys
from components.models import ObservationRatio, InfectRatio, TouchRatio, DeadRatio, DummyModel, IsolationRatio
import pandas as pd
import json
from components.utils import prepareData, codeDict, flowHubei, flowOutData, get_data_path, get_important_date, get_seed_num, get_core_num, clear_child_process, it_code_dict, construct_x, format_out
from components.simulator import Simulator, InfectionList, get_loss, get_newly_loss
from show_simulation_result import run_simulation
import numpy as np
from show_simulation_result import plot1, plot1_shade
from res_analysis import load_and_save

sys.path.append(os.path.join(os.path.dirname(__file__), "../ncovmodel"))
training_end_date = date(2020, 2, 17)
vali_end_data = date(2020, 3, 22)
def run_opt(city, budget, start_date, important_dates, infectratio_range=None,
            dummy_range=None, unob_flow_num=None, repeat_time=1, init_samples=None,training_date_end=None,
            json_name='data_run.json', seed=1, loss_ord=0.0,
            unob_period=None, obs_period=None, iso_period=None, cure_period=None,
            isoratio_it=None):

    if city == 420000:
        infectratio_range = [0., 0.05]
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
    touchratio = TouchRatio(1, [[0.0, 0.6]], [True])
    touchratiointra = TouchRatio(1, [[0, 1]], [True])
    obs = ObservationRatio(1, [[0.0, 0.3]], [True])
    dead = DeadRatio(1, [[0., 0.01]], [True])
    if isoratio_it is None:
        isoratio = IsolationRatio(1, [[0.2, 0.5]], [True])
    else:
        isoratio = IsolationRatio(1, [isoratio_it], [True])

    dummy = DummyModel(1, [dummy_range], [True, True])
    cure_ratio = InfectRatio(1, [[0., 0.1]], [True])


    # set the time of applying touchratio
    simulator = Simulator(city, infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra, cure_ratio, important_dates,
                          unob_flow_num=unob_flow_num, flow_out_data=flow_out_data, training_date_end=training_date_end)
    # set period here
    simulator.set_period()
    test_date = datetime.strptime(history_real['date'].max(), '%Y-%m-%d').date() - timedelta(days_predict)
    history_real = history_real[history_real['adcode'] == city]
    history_real = history_real[history_real['date'] >= str(start_date)]
    history_train = history_real[history_real['date'] <= str(test_date)]

    x, y = simulator.fit(history_train, budget=budget, server_num=get_core_num(),
                         repeat=repeat_time, seed=seed, intermediate_freq=10000, init_samples=init_samples, loss_ord=loss_ord)

    print('best_solution: x = ', x, 'y = ', y)
    simulator.set_param(x)
    run_simulation(x, city, 60, 60, start_date, get_important_date(city), unob_flow_num=unob_flow_num, json_name=json_name)
    duration = len(real_data["date"].unique()) - 1
    sim_res, _ = simulator.simulate(str(start_date), duration)
    print('RMSE: ', get_newly_loss(sim_res, history_real))
    return x, sim_res





def disp2():
    load_and_save('data_run_no_healed{}.json', 'data_run_no_healed_0.1_{}.json', 10, 0)
    load_and_save('data_run_no_healed{}.json', 'data_run_no_healed_0.2_{}.json', 10, 10)
    load_and_save('data_run_no_healed{}.json', 'data_run_no_healed_0.3_{}.json', 10, 20)
    data_01 = json.load(open('data_run_no_healed_0.1_middle.json', 'r'))
    min_data_01 = json.load(open('data_run_no_healed_0.1_xmin.json', 'r'))
    max_data_01 = json.load(open('data_run_no_healed_0.1_xmax.json', 'r'))
    x_01 = construct_x(data_01)
    min_x_01 = construct_x(min_data_01)
    max_x_01 = construct_x(max_data_01)

    data_02 = json.load(open('data_run_no_healed_0.2_middle.json', 'r'))
    min_data_02 = json.load(open('data_run_no_healed_0.2_xmin.json', 'r'))
    max_data_02 = json.load(open('data_run_no_healed_0.2_xmax.json', 'r'))

    x_02 = construct_x(data_02)
    min_x_02 = construct_x(min_data_02)
    max_x_02 = construct_x(max_data_02)

    x_01.append(x_02[0])
    min_x_01.append(min_x_02[0])
    max_x_01.append(max_x_02[0])

    data_04 = json.load(open('data_run_no_healed_0.3_middle.json', 'r'))
    min_data_04 = json.load(open('data_run_no_healed_0.3_xmin.json', 'r'))
    max_data_04 = json.load(open('data_run_no_healed_0.3_xmax.json', 'r'))

    x_04 = construct_x(data_04)
    min_x_04 = construct_x(min_data_04)
    max_x_04 = construct_x(max_data_04)

    x_01.append(x_04[0])
    min_x_01.append(min_x_04[0])
    max_x_01.append(max_x_04[0])

    data_03 = json.load(open('data_run_middle.json', 'r'))
    min_data_03 = json.load(open('data_run_xmin.json', 'r'))
    max_data_03 = json.load(open('data_run_xmax.json', 'r'))

    x_03 = construct_x(data_03, ['420000'])
    min_x_03 = construct_x(min_data_03, ['420000'])
    max_x_03 = construct_x(max_data_03, ['420000'])

    x_01.insert(0, x_03[0])
    min_x_01.insert(0, min_x_03[0])
    max_x_01.insert(0, max_x_03[0])



    city = 420000
    plot1_shade(data_01['real_confirmed'][str(city)],
                data_01['sim_confirmed'][str(city)],
                min_data_01['sim_confirmed'][str(city)],
                max_data_01['sim_confirmed'][str(city)],
                 '', 'no_heal_01_sim_real.pdf', 7)
    print('0.1: {:.3f} ({:.3f}-{:.3f})'.format(data_01['newly_confirmed_loss'][str(city)],
                                          min_data_01['newly_confirmed_loss'][str(city)],
                                          max_data_01['newly_confirmed_loss'][str(city)],
                                          ))
    plot1_shade(data_02['real_confirmed'][str(city)],
                data_02['sim_confirmed'][str(city)],
                min_data_02['sim_confirmed'][str(city)],
                max_data_02['sim_confirmed'][str(city)],
                '', 'no_heal_02_sim_real.pdf', 7)

    print('0.2: {:.3f} ({:.3f}-{:.3f})'.format(data_02['newly_confirmed_loss'][str(city)],
                                          min_data_02['newly_confirmed_loss'][str(city)],
                                          max_data_02['newly_confirmed_loss'][str(city)],
                                          ))
    plot1_shade(data_04['real_confirmed'][str(city)],
                data_04['sim_confirmed'][str(city)],
                min_data_04['sim_confirmed'][str(city)],
                max_data_04['sim_confirmed'][str(city)],
                '', 'no_heal_03_sim_real.pdf', 7)
    print(x_01)
    print(x_02)
    format_out(x_01, min_x_01, max_x_01)



def disp():
    flow_out_data = flowOutData()
    for i in range(4):
        real_data = pd.read_csv(get_data_path())
        data = json.load(open('data_run_no_healed{}.json'.format(int(i))))
        infectratio = InfectRatio(1, [[0.,1.]], [True])
        touchratio = TouchRatio(1, [[0.0, 0.6]], [True])
        touchratiointra = TouchRatio(1, [[0, 1]], [True])
        obs = ObservationRatio(1, [[0.0, 0.3]], [True])
        dead = DeadRatio(1, [[0., 0.01]], [True])
        isoratio = IsolationRatio(1, [[0.0, 0.5]], [True])
        dummy = DummyModel(1, [[0, 400]], [True, True])
        cure_ratio = InfectRatio(1, [[0., 0.1]], [True])
        city = 420000
        start_date = date(2020,1,11)
        # set the time of applying touchratio
        simulator = Simulator(city, infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra, cure_ratio,
                              [get_important_date(city)],
                              unob_flow_num=None, flow_out_data=flow_out_data,
                              training_date_end=None)
        simulator.set_param(data['x'][str(city)])
        duration = len(real_data["date"].unique()) - 1
        sim_res, _ = simulator.simulate(str(start_date), duration)
        plot1(data['real_confirmed'][str(city)], sim_res['observed'], '', 'no_heal{}.pdf'.format(int(i)), 7)

def main():
    training_end_date = None
    for i in range(30):
        iso_ratio_it = [0.1, 0.5]
        if i >= 10:
            iso_ratio_it = [0.2,0.5]
        if i >= 20:
            iso_ratio_it = [0.3, 0.5]
        x = run_opt(420000, 200000, start_date=date(2020, 1, 11), important_dates=[get_important_date(420000)],
                repeat_time=1, training_date_end=training_end_date, isoratio_it=iso_ratio_it, seed=1, json_name='data_run_no_healed{}.json'.format(int(i)))

        clear_child_process()


if __name__ == '__main__':
    main()
    #disp()
    disp2()
