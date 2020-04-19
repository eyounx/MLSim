from show_simulation_foreign import run_simulation2
from run import initHubei, run_opt
from components.utils import prepareData, codeDict, flowHubei, flowOutData, get_data_path, get_important_date, get_seed_num
import os
from datetime import date
import pandas as pd
import numpy as np
import json
from res_analysis import  res_analysis, load_and_save
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load_inter', type=bool, default=False)
    args = parser.parse_args()
    return args


def get_params(total_params, city):
    return list(total_params[str(city)].values)


def main(load_inter_flag=False):
    if not load_inter_flag:
        real_data = list(pd.read_csv(get_data_path(True))['adcode'].unique())
        start_date = date(2020, 1, 24)
        real_data1 = pd.read_csv(get_data_path(True))
        history_real = prepareData(real_data1)
        rerun_cities = None
        for i in range(get_seed_num()):
            total_params = pd.read_csv(os.path.join(os.path.dirname(__file__), './params_foreign{}.csv'.format(int(i))))
            for ind, city in enumerate(real_data):
                print(i, ind, city, str(get_important_date(city)))
                x = get_params(total_params, city)
                if rerun_cities is None or city in rerun_cities:
                    run_simulation2(x, city, 60, 60, start_date, get_important_date(city), history_real,
                            unob_flow_num=None, json_name='data_run_foreign{}.json'.format(int(i)))
    # choose the final parameters
    max_data, min_data, mean_data, best_data, \
    std_data, middle_data, xmin_data, xmax_data, \
    xmean_data, xstd_data = load_and_save('./data_run_foreign{}.json', './data_run_foreign_{}.json')

    """
    it_total_params = mean_data['x']
    for ind, city in enumerate(real_data):
        print('final: ', ind, city)
        x = it_total_params[str(city)]
        run_simulation2(x, city, 60, 60, start_date, get_important_date(city), history_real,
                        unob_flow_num=None, json_name='data_run_foreign_mean_deduction.json')
    it_total_params = middle_data['x']
    for ind, city in enumerate(real_data):
        print('final: ', ind, city)
        x = it_total_params[str(city)]
        run_simulation2(x, city, 60, 60, start_date, get_important_date(city), history_real,
                        unob_flow_num=None, json_name='data_run_foreign_middle_deduction.json')
    """

if __name__ == '__main__':
    args = parse_args()
    main(args.load_inter)


