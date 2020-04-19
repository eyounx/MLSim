from show_simulation_result import run_simulation2
from run import initHubei
from components.utils import prepareData, flowHubei, get_data_path, get_important_date, get_seed_num, append_to_json
import os
from datetime import date
import pandas as pd
import json
import numpy as np
from res_analysis import res_analysis, load_and_save
import argparse


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load_inter', type=bool, default=False)
    args = parser.parse_args()
    return args

def get_params(total_params, city):
    return list(total_params[str(city)].values)

def get_total_province(it_path):
    with open(it_path, 'r') as f:
        data = json.load(f)
    for key in data.keys():
        sum_list = None
        for subkey in data[key]:
            if len(subkey) >= 5:
                if isinstance(data[key][subkey], list):
                    if sum_list is None:
                        sum_list = data[key][subkey].copy()
                    else:
                        for i in range(len(sum_list)):
                            sum_list[i] += data[key][subkey][i]
                else:
                    if sum_list is None:
                        sum_list = data[key][subkey]
                    else:
                        sum_list += data[key][subkey]
        append_to_json(it_path, key, '0000', sum_list)

    with open(it_path, 'r') as f:
        data = json.load(f)
    lreal = len(data['real_confirmed']['0000'])
    c_con = list(data['sim_cum_confirmed_deduction_s1']['0000'])[lreal-1]
    c_inf = list(data['sim_cum_infection_deduction_s1']['0000'])[lreal - 1]
    append_to_json(it_path, 'current_asym', '0000', c_con / c_inf)
    c_con = list(data['sim_cum_confirmed_deduction_s1']['0000'])[-1]
    c_inf = list(data['sim_cum_infection_deduction_s1']['0000'])[-1]
    append_to_json(it_path, 'final_asym', '0000', c_con / c_inf)

    sum_list = []
    for i in range(len(data['sim_cum_infection_deduction_s1']['0000'])):
        sum_list.append(data['sim_cum_infection_deduction_s1']['0000'][i] - data['sim_cum_confirmed_deduction_s1']['0000'][i])
    append_to_json(it_path, 'sim_cum_infe_minus_conf_s1', '0000', sum_list)



def prepare_data_china(load_inter_flag=False):
    real_data = pd.read_csv(get_data_path())['adcode'].unique()
    rerun_cities = None
    if rerun_cities is not None:
        real_data = rerun_cities.copy()
    start_date = date(2020, 1, 11)
    real_data1 = pd.read_csv(get_data_path())
    history_real = prepareData(real_data1)
    if not load_inter_flag:
        print('re-simulate the progress')
        province_travel_dict = flowHubei()
        for i in range(get_seed_num()):
            total_params = pd.read_csv(os.path.join(os.path.dirname(__file__), './params{}.csv'.format(int(i))))
            unob_flow_num = initHubei(get_params(total_params, 420000), start_date,
                                      important_date=[get_important_date(420000)],
                                      travel_from_hubei=province_travel_dict)
            for ind, city in enumerate(real_data):
                print(i, ind, city)
                x = get_params(total_params, city)
                run_simulation2(x, city, 90, 90, start_date, get_important_date(city), history_real,
                                unob_flow_num=unob_flow_num, json_name='data_run{}.json'.format(int(i)))
    else:
        print('directly load the pre-simulated data')
    for i in range(get_seed_num()):
        get_total_province('data_run{}.json'.format(int(i)))

    # choose the final parameters
    max_data, min_data, mean_data, best_data, std_data, middle_data, \
    xmin_data, xmax_data, xmean_data, xstd_data = load_and_save(
        './data_run{}.json', './data_run_{}.json')

    """
    it_total_params = mean_data['x']
    unob_flow_num = initHubei(it_total_params['420000'], start_date,
                              important_date=[get_important_date(420000)],
                              travel_from_hubei=province_travel_dict)
    for ind, city in enumerate(real_data):
        print('final: ', ind, city)
        x = it_total_params[str(city)]
        run_simulation2(x, city, 90, 90, start_date, get_important_date(city), history_real,
                        unob_flow_num=unob_flow_num, json_name='data_run_mean_deduction.json')
    it_total_params = middle_data['x']
    unob_flow_num = initHubei(it_total_params['420000'], start_date,
                              important_date=[get_important_date(420000)],
                              travel_from_hubei=province_travel_dict)
    for ind, city in enumerate(real_data):
        print('final: ', ind, city)
        x = it_total_params[str(city)]
        run_simulation2(x, city, 90, 90, start_date, get_important_date(city), history_real,
                        unob_flow_num=unob_flow_num, json_name='data_run_middle_deduction.json')
    """

if __name__ == '__main__':
    args = parse_args()
    prepare_data_china(args.load_inter)


