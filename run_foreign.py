from datetime import date, timedelta, datetime
import os, sys
from components.models import ObservationRatio, InfectRatio, TouchRatio, DeadRatio, DummyModel, IsolationRatio
import pandas as pd
from components.utils import prepareData, get_data_path, get_important_date, get_seed_num, it_code_dict, get_core_num, clear_child_process
from components.simulator import Simulator
from show_simulation_foreign import run_simulation

sys.path.append(os.path.join(os.path.dirname(__file__), "../ncovmodel"))


def run_opt(city, budget, start_date, important_dates, infectratio_range=None,
            dummy_range=None, unob_flow_num=None, repeat_time=1, init_samples=None,
            training_end_date=None, json_name='data_run.json', seed=3, loss_ord=0.0,
            touch_range=None,iso_range=None):
    assert infectratio_range is not None and dummy_range is not None
    days_predict = 0
    # load data
    real_data = pd.read_csv(get_data_path(True))
    history_real = prepareData(real_data)
    flow_out_data = None
    # initialize models

    infectratio = InfectRatio(1, [infectratio_range], [True])
    if touch_range is None:
        touchratio = TouchRatio(1, [[0.999, 1.0000]], [True])
    else:
        touchratio = TouchRatio(1, [touch_range], [True])
    touchratiointra = TouchRatio(1, [[0, 1]], [True])
    obs = ObservationRatio(1, [[0.0, 0.3]], [True])
    dead = DeadRatio(1, [[0., 0.01]], [True])
    if iso_range is None:
        isoratio = IsolationRatio(1, [[0.03, 0.12]], [True])
    else:
        isoratio = IsolationRatio(1, [iso_range], [True])

    dummy = DummyModel(1, [dummy_range], [True, True])
    cure_ratio = InfectRatio(1, [[0., 0.15]], [True])

    simulator = Simulator(city, infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra, cure_ratio, important_dates,
                          unob_flow_num=unob_flow_num, flow_out_data=flow_out_data, training_date_end=training_end_date)
    test_date = datetime.strptime(history_real['date'].max(), '%Y-%m-%d').date() - timedelta(days_predict)
    history_real = history_real[history_real['adcode'] == city]
    history_real = history_real[history_real['date'] >= str(start_date)]
    history_train = history_real[history_real['date'] <= str(test_date)]

    x, y = simulator.fit(history_train, budget=budget, server_num=get_core_num(),
                         repeat=repeat_time, seed=seed, intermediate_freq=10000, init_samples=init_samples,
                         loss_ord=loss_ord)
    print('best_solution: x = ', x, 'y = ', y)
    simulator.set_param(x)
    run_simulation(x, city, 60, 60, start_date, get_important_date(city), unob_flow_num=unob_flow_num, json_name=json_name)
    return x


if __name__ == '__main__':
    real_data = list(pd.read_csv(get_data_path(True))['adcode'].unique())
    province_code_dict = it_code_dict()
    rerun_list = None
    old_param_buffer = None
    training_end_date = None
    if rerun_list is not None:
        old_param_buffer = [pd.read_csv('params_foreign{}.csv'.format(int(i))) for i in range(get_seed_num())]
    for i in range(get_seed_num()):
        all_param = {}
        for ind, item in enumerate(real_data):
            print(i, ind, item, province_code_dict[item])
            if rerun_list is None or item in rerun_list:
                touch_range = [0, 0.33333]
                iso_range = None
                x = run_opt(item, 80000, start_date=date(2020, 1, 24), important_dates=[get_important_date(item)],
                            infectratio_range=[0.0, 0.05], dummy_range=[0, 400], unob_flow_num=None, repeat_time=1,
                            seed=i, json_name='data_run_foreign{}.json'.format(int(i)), loss_ord=1.5, touch_range=touch_range,
                            iso_range=iso_range, training_end_date=training_end_date)
                all_param[item] = x
                clear_child_process()
            else:
                all_param[item] = list(old_param_buffer[i][str(item)])
        all_param_df = pd.DataFrame(all_param)
        all_param_df.to_csv('params_foreign{}.csv'.format(int(i)), index=False)

