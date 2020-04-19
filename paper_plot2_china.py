from show_simulation_result import plot1, plot11, plot12, plot33, plot33_time, plot1_shade, plot33_shade, plot33_time_shade, plot12_shade
from components.utils import get_data_path, it_code_dict, get_R0, get_DT, construct_x, format_out
import os
from datetime import date, timedelta
import pandas as pd
import numpy as np
import json



def get_total(data, type, cities):
    l = len(data[type][str(cities[0])])
    res = l * [0]
    for ind, city in enumerate(cities):
        for i in range(l):
            res[i] += data[type][str(city)][i]
    return res

today = date(2020, 4,15)

def main():
    data = {}
    code_dict = it_code_dict()
    fmt = 'pdf'
    total_cities = pd.read_csv(get_data_path())['adcode'].unique()
    #important_cities = [420000,110000,120000,320000,330000,440000,510000]
    important_cities = total_cities
    with open('./data_run_middle.json', 'r') as f:
        data = json.load(f)
    with open('./data_run_xmin.json', 'r') as f:
        min_data = json.load(f)
    with open('./data_run_xmax.json', 'r') as f:
        max_data = json.load(f)
    if not os.path.exists('./img'):
        os.mkdir('./img')
    if not os.path.exists('./img/0000'):
        os.makedirs('./img/0000')
    for item in important_cities:
        it_path = './img/{}'.format(str(item))
        if not os.path.exists(it_path):
            os.mkdir(it_path)
    # test


    # end test
    cum_dead_res = [get_total(data, 'real_cum_dead', total_cities)[-1],
                    get_total(data, 'sim_cum_dead_s1', total_cities)[-1],
                    get_total(data, 'sim_cum_dead_s2', total_cities)[-1],
                    get_total(data, 'sim_cum_dead_s3', total_cities)[-1],
                    get_total(data, 'sim_cum_dead_s4', total_cities)[-1],
                    get_total(data, 'sim_cum_dead_s5', total_cities)[-1],
                    get_total(data, 'sim_cum_dead_s6', total_cities)[-1],
                    get_total(data, 'sim_cum_dead_s7', total_cities)[-1],
                    get_total(data, 'sim_cum_dead_s8', total_cities)[-1],
                    ]
    d_cum =get_total(data, 'sim_cum_dead_s1', total_cities)[-60:]
    print(d_cum[0], d_cum[-1], d_cum[-1]-d_cum[0])
    for i in range(2, len(cum_dead_res)):
        cum_dead_res[i] = cum_dead_res[i] + cum_dead_res[0] - cum_dead_res[1] + (d_cum[-1]-d_cum[0])
    cum_dead_res[1] = cum_dead_res[1] + cum_dead_res[0] - cum_dead_res[1] + (d_cum[-1]-d_cum[0])
    print('cum dead res: ',cum_dead_res)
    cum_confirm_res = [get_total(data, 'real_cum_confirmed', total_cities)[-1],
                    get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities)[-1],
                    get_total(data, 'sim_cum_confirmed_deduction_s2', total_cities)[-1],
                    get_total(data, 'sim_cum_confirmed_deduction_s3', total_cities)[-1],
                    get_total(data, 'sim_cum_confirmed_deduction_s4', total_cities)[-1],
                    get_total(data, 'sim_cum_confirmed_deduction_s5', total_cities)[-1],
                    get_total(data, 'sim_cum_confirmed_deduction_s6', total_cities)[-1],
                    get_total(data, 'sim_cum_confirmed_deduction_s7', total_cities)[-1],
                       get_total(data, 'sim_cum_confirmed_deduction_s8', total_cities)[-1],
                       ]
    print('cum confirm case: ', cum_confirm_res)
    cum_res = [cum_confirm_res[1],
               cum_confirm_res[8],
                       cum_confirm_res[3],
                       cum_confirm_res[2],
                       cum_confirm_res[1],
                       cum_confirm_res[4],
                       cum_confirm_res[5],
                       cum_confirm_res[6],
                       cum_dead_res[1],
                        cum_dead_res[8],
                       cum_dead_res[3],
                       cum_dead_res[2],
                       cum_dead_res[1],
                       cum_dead_res[4],
                       cum_dead_res[5],
                       cum_dead_res[6]]
    for item in cum_res:
        print(int(item),  ' & ', end='')
    print('\\\\')
    cum_infection_res = [np.sum(get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[-1]),
                         np.sum(get_total(data, 'sim_cum_infection_deduction_s2', total_cities)[-1]),
                         np.sum(get_total(data, 'sim_cum_infection_deduction_s3', total_cities)[-1]),
                         np.sum(get_total(data, 'sim_cum_infection_deduction_s4', total_cities)[-1]),
                         np.sum(get_total(data, 'sim_cum_infection_deduction_s5', total_cities)[-1]),
                         np.sum(get_total(data, 'sim_cum_infection_deduction_s6', total_cities)[-1]),
                         np.sum(get_total(data, 'sim_cum_infection_deduction_s7', total_cities)[-1]),
                        ]
    print(cum_infection_res)
    lreal = len(get_total(data, 'real_cum_confirmed', total_cities))
    cum_infection_res = [np.sum(get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[lreal-1]),
                         np.sum(get_total(data, 'sim_cum_infection_deduction_s2', total_cities)[lreal-1]),
                         np.sum(get_total(data, 'sim_cum_infection_deduction_s3', total_cities)[lreal-1]),
                         np.sum(get_total(data, 'sim_cum_infection_deduction_s4', total_cities)[lreal-1]),
                         np.sum(get_total(data, 'sim_cum_infection_deduction_s5', total_cities)[lreal-1]),
                         np.sum(get_total(data, 'sim_cum_infection_deduction_s6', total_cities)[lreal-1]),
                         np.sum(get_total(data, 'sim_cum_infection_deduction_s7', total_cities)[lreal-1]),
                         ]
    print(cum_infection_res)
    print(get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[lreal-1],
          get_total(data, 'sim_cum_confirmed', total_cities)[lreal - 1],
          get_total(data, 'real_cum_confirmed', total_cities)[lreal - 1],
          )
    print(get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[- 1],
          get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities)[- 1],
          get_total(data, 'real_cum_confirmed', total_cities)[- 1],
          )
    print("final day: ", date(2020,1,11) + timedelta(days=len(get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities))))
    final_date = date(2020,1,11) + timedelta(days=len(get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities)))
    cur_date = date(2020,1, 11) + timedelta(days=len(get_total(data, 'real_confirmed', total_cities)))
    # draw compare to real data for every city and full
    print('drawing real/sim compare')
    print('city', 'real_confirmed', 'sim_confirmed', 'sim_infect', 'sim_confirmed / sim_infect')
    print('Province&real confirmed&MLSim confirmed&MLSim infected& proportion of asymptomatic\\\\\\hline')
    for item in total_cities:
        real_cum = data['real_cum_confirmed'][str(item)][-1]
        sim_cum_confirm =data['sim_cum_confirmed'][str(item)][-1]
        sim_cum_infect = data['sim_cum_infection_deduction_s1'][str(item)][lreal-1]
        #print(code_dict[int(item)],
        #      real_cum,
        #      sim_cum_confirm,
        #      sim_cum_infect,
        #      #np.round(real_cum / sim_cum_infect, 3),
        #      np.round(sim_cum_confirm / sim_cum_infect, 3),sep='\t\t')
        print(code_dict[int(item)], ' & ',
              real_cum, ' & ',
              int(sim_cum_confirm), ' & ',
              int(sim_cum_infect), ' & ',
              np.round((1-sim_cum_confirm / sim_cum_infect)*100, 3), '$\\%$\\\\ ',sep='')
    print('China\'s mainland', ' & ',
          int(get_total(data, 'real_cum_confirmed', total_cities)[-1]), ' & ',
          int(get_total(data, 'sim_cum_confirmed', total_cities)[-1]), ' & ',
          int(get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[lreal-1]), ' & ',
          np.round((1 - (get_total(data, 'sim_cum_confirmed', total_cities)[-1]) / (get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[lreal-1])) * 100, 3), '$\\%$\\\\ ', sep='')

    diff_time_confirm = [
        int(get_total(data,'sim_cum_confirmed_deduction_s1',total_cities)[-1]),
        int(get_total(data, 'sim_cum_confirmed_deduction_s9', total_cities)[-1]),
        int(get_total(data, 'sim_cum_confirmed_deduction_s10', total_cities)[-1]),
        int(get_total(data, 'sim_cum_confirmed_deduction_s11', total_cities)[-1]),
        int(get_total(data, 'sim_cum_confirmed_deduction_s12', total_cities)[-1]),
        int(get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[-1]),
        int(get_total(data, 'sim_cum_infection_deduction_s9', total_cities)[-1]),
        int(get_total(data, 'sim_cum_infection_deduction_s10', total_cities)[-1]),
        int(get_total(data, 'sim_cum_infection_deduction_s11', total_cities)[-1]),
        int(get_total(data, 'sim_cum_infection_deduction_s12', total_cities)[-1]),
    ]
    print('diff confirm: ', diff_time_confirm)
    for item in diff_time_confirm:
        print(int(item),  ' & ', end='')
    print('\\\\')
    for item in total_cities:
        plot1_shade(data['real_cum_confirmed'][str(item)],
                    data['sim_cum_confirmed'][str(item)],
                    min_data['sim_cum_confirmed'][str(item)],
                    max_data['sim_cum_confirmed'][str(item)],
                    code_dict[int(item)],
                    'img/{}/cum_real_sim.{}'.format(str(item), fmt))

        plot1_shade(data['real_confirmed'][str(item)],
                    data['sim_confirmed'][str(item)],
                    min_data['sim_confirmed'][str(item)],
                    max_data['sim_confirmed'][str(item)],
                    code_dict[int(item)],
                    'img/{}/increase_real_sim.{}'.format(str(item), fmt))

        plot1_shade(data['real_cum_confirmed'][str(item)],
                    data['sim_cum_confirmed_deduction_s1'][str(item)],
                    min_data['sim_cum_confirmed_deduction_s1'][str(item)],
                    max_data['sim_cum_confirmed_deduction_s1'][str(item)],
                    code_dict[int(item)],
                    'img/{}/cum_real_sim_deduction.{}'.format(str(item), fmt),interval=20)

        plot1_shade(data['real_confirmed'][str(item)],
                    data['sim_confirmed_deduction_s1'][str(item)],
                    min_data['sim_confirmed_deduction_s1'][str(item)],
                    max_data['sim_confirmed_deduction_s1'][str(item)],
                    code_dict[int(item)],
                    'img/{}/increase_real_sim_dedection.{}'.format(str(item), fmt), interval=20)

        plot33_shade(data['sim_cum_confirmed_deduction_s1'][str(item)], min_data['sim_cum_confirmed_deduction_s1'][str(item)], max_data['sim_cum_confirmed_deduction_s1'][str(item)],
                     data['sim_cum_confirmed_deduction_s2'][str(item)], min_data['sim_cum_confirmed_deduction_s2'][str(item)], max_data['sim_cum_confirmed_deduction_s2'][str(item)],
                     data['sim_cum_confirmed_deduction_s3'][str(item)], min_data['sim_cum_confirmed_deduction_s3'][str(item)], max_data['sim_cum_confirmed_deduction_s3'][str(item)],
                     data['sim_cum_confirmed_deduction_s8'][str(item)], min_data['sim_cum_confirmed_deduction_s8'][str(item)], max_data['sim_cum_confirmed_deduction_s8'][str(item)],
                     data['real_cum_confirmed'][str(item)],
                     code_dict[int(item)],
                     'img/{}/cum_confirmed_prediction.{}'.format(str(item), fmt),
                     touchratio=data['x'][str(item)][1],
                     ratio_low=data['touch_ratio_low'][str(item)],
                     ratio_high=data['touch_ratio_hight'][str(item)],
                     loc='lower right',
                     date_it=date(2020, 1, 23),
                     ext_flag=True
                     )

        plot33_shade(data['sim_confirmed_deduction_s1'][str(item)], min_data['sim_confirmed_deduction_s1'][str(item)], max_data['sim_confirmed_deduction_s1'][str(item)],
                     data['sim_confirmed_deduction_s2'][str(item)], min_data['sim_confirmed_deduction_s2'][str(item)], max_data['sim_confirmed_deduction_s2'][str(item)],
                     data['sim_confirmed_deduction_s3'][str(item)],min_data['sim_confirmed_deduction_s3'][str(item)], max_data['sim_confirmed_deduction_s3'][str(item)],
                     data['sim_confirmed_deduction_s8'][str(item)], min_data['sim_confirmed_deduction_s8'][str(item)], max_data['sim_confirmed_deduction_s8'][str(item)],
                     data['real_confirmed'][str(item)],
                     code_dict[int(item)],
                     'img/{}/confirmed_prediction.{}'.format(str(item), fmt),
                     touchratio=data['x'][str(item)][1],
                     ratio_low=data['touch_ratio_low'][str(item)],
                     ratio_high=data['touch_ratio_hight'][str(item)],
                     loc='upper right',
                     date_it=date(2020, 1, 23),
                     ext_flag=True
                     )

    plot33_shade(
        get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities), get_total(min_data,'sim_cum_confirmed_deduction_s1',total_cities),  get_total(max_data,'sim_cum_confirmed_deduction_s1',total_cities),
        get_total(data, 'sim_cum_confirmed_deduction_s2', total_cities), get_total(min_data,'sim_cum_confirmed_deduction_s2',total_cities),  get_total(max_data,'sim_cum_confirmed_deduction_s2',total_cities),
        get_total(data, 'sim_cum_confirmed_deduction_s3', total_cities), get_total(min_data,'sim_cum_confirmed_deduction_s3',total_cities),  get_total(max_data,'sim_cum_confirmed_deduction_s3',total_cities),
        get_total(data, 'sim_cum_confirmed_deduction_s8', total_cities), get_total(min_data,'sim_cum_confirmed_deduction_s8',total_cities),  get_total(max_data,'sim_cum_confirmed_deduction_s8',total_cities),
        get_total(data,'real_cum_confirmed', total_cities),
        'China\'s mainland',
        'img/{}/cum_confirmed_prediction.{}'.format('0000', fmt),
        touchratio=0.1,
        ratio_low=0.2,
        ratio_high=0.4,
        loc='lower right',
        date_it=date(2020, 1, 23),
        ext_flag=True
    )

    plot33_shade(

        get_total(data, 'sim_confirmed_deduction_s1', total_cities), get_total(min_data,'sim_confirmed_deduction_s1',total_cities),  get_total(max_data,'sim_confirmed_deduction_s1',total_cities),
        get_total(data, 'sim_confirmed_deduction_s2', total_cities), get_total(min_data,'sim_confirmed_deduction_s2',total_cities),  get_total(max_data,'sim_confirmed_deduction_s2',total_cities),
        get_total(data, 'sim_confirmed_deduction_s3', total_cities), get_total(min_data,'sim_confirmed_deduction_s3',total_cities),  get_total(max_data,'sim_confirmed_deduction_s3',total_cities),
        get_total(data, 'sim_confirmed_deduction_s8', total_cities), get_total(min_data,'sim_confirmed_deduction_s8',total_cities),  get_total(max_data,'sim_confirmed_deduction_s8',total_cities),
        get_total(data,'real_confirmed', total_cities),
        'China\'s mainland',
        'img/{}/confirmed_prediction.{}'.format('0000', fmt),
        touchratio=0.1,
        ratio_low=0.2,
        ratio_high=0.3,
        loc='upper right',
        date_it=date(2020, 1, 23),
        ext_flag=True
    )
    print('==================================', str(final_date))
    print('Jan 23 different control measure: ')
    print('confirmed')
    print('middle: ')
    print('100\%', format(int(get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities)[-1]), ','),
          '190\%', format(int(get_total(data, 'sim_cum_confirmed_deduction_s2', total_cities)[-1]), ','),
          '160\%', format(int(get_total(data, 'sim_cum_confirmed_deduction_s3', total_cities)[-1]), ','),
          '130\%', format(int(get_total(data, 'sim_cum_confirmed_deduction_s8', total_cities)[-1]), ','), )
    print('min: ')
    print('100\%', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s1', total_cities)[-1]), ','),
          '190\%', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s2', total_cities)[-1]), ','),
          '160\%', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s3', total_cities)[-1]), ','),
          '130\%', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s8', total_cities)[-1]), ','), )
    print('max:')
    print('100\%', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s1', total_cities)[-1]), ','),
          '190\%', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s2', total_cities)[-1]), ','),
          '160\%', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s3', total_cities)[-1]), ','),
          '130\%', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s8', total_cities)[-1]), ','), )
    print('infected')
    print('middle: ')
    print('100\%', format(int(get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[-1]), ','),
          '190\%', format(int(get_total(data, 'sim_cum_infection_deduction_s2', total_cities)[-1]), ','),
          '160\%', format(int(get_total(data, 'sim_cum_infection_deduction_s3', total_cities)[-1]), ','),
          '130\%', format(int(get_total(data, 'sim_cum_infection_deduction_s8', total_cities)[-1]), ','), )
    print('min: ')
    print('100\%', format(int(get_total(min_data, 'sim_cum_infection_deduction_s1', total_cities)[-1]), ','),
          '190\%', format(int(get_total(min_data, 'sim_cum_infection_deduction_s2', total_cities)[-1]), ','),
          '160\%', format(int(get_total(min_data, 'sim_cum_infection_deduction_s3', total_cities)[-1]), ','),
          '130\%', format(int(get_total(min_data, 'sim_cum_infection_deduction_s8', total_cities)[-1]), ','), )
    print('max:')
    print('100\%', format(int(get_total(max_data, 'sim_cum_infection_deduction_s1', total_cities)[-1]), ','),
          '190\%', format(int(get_total(max_data, 'sim_cum_infection_deduction_s2', total_cities)[-1]), ','),
          '160\%', format(int(get_total(max_data, 'sim_cum_infection_deduction_s3', total_cities)[-1]), ','),
          '130\%', format(int(get_total(max_data, 'sim_cum_infection_deduction_s8', total_cities)[-1]), ','), )
    print('==================================', str(final_date))

    plot33_shade(
        get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities),
        get_total(min_data, 'sim_cum_confirmed_deduction_s1', total_cities),
        get_total(max_data, 'sim_cum_confirmed_deduction_s1', total_cities),
        get_total(data, 'sim_cum_confirmed_deduction_s5', total_cities),
        get_total(min_data, 'sim_cum_confirmed_deduction_s5', total_cities),
        get_total(max_data, 'sim_cum_confirmed_deduction_s5', total_cities),
        get_total(data, 'sim_cum_confirmed_deduction_s4', total_cities),
        get_total(min_data, 'sim_cum_confirmed_deduction_s4', total_cities),
        get_total(max_data, 'sim_cum_confirmed_deduction_s4', total_cities),
        get_total(data, 'sim_cum_confirmed_deduction_s6', total_cities),
        get_total(min_data, 'sim_cum_confirmed_deduction_s6', total_cities),
        get_total(max_data, 'sim_cum_confirmed_deduction_s6', total_cities),
        get_total(data, 'real_cum_confirmed', total_cities),
        'China\'s mainland',
        'img/{}/cum_confirmed_prediction_simdate.{}'.format('0000', fmt),
        touchratio=0.1,
        ratio_low=0.2,
        ratio_high=0.4,
        loc='lower right',
        date_it=date(2020, 3, 1),
    )

    plot33_shade(
        get_total(data, 'sim_confirmed_deduction_s1', total_cities),
        get_total(min_data, 'sim_confirmed_deduction_s1', total_cities),
        get_total(max_data, 'sim_confirmed_deduction_s1', total_cities),
        get_total(data, 'sim_confirmed_deduction_s5', total_cities),
        get_total(min_data, 'sim_confirmed_deduction_s5', total_cities),
        get_total(max_data, 'sim_confirmed_deduction_s5', total_cities),
        get_total(data, 'sim_confirmed_deduction_s4', total_cities),
        get_total(min_data, 'sim_confirmed_deduction_s4', total_cities),
        get_total(max_data, 'sim_confirmed_deduction_s4', total_cities),
        get_total(data, 'sim_confirmed_deduction_s6', total_cities),
        get_total(min_data, 'sim_confirmed_deduction_s6', total_cities),
        get_total(max_data, 'sim_confirmed_deduction_s6', total_cities),
        get_total(data, 'real_confirmed', total_cities),
        'China\'s mainland',
        'img/{}/confirmed_prediction_simdate.{}'.format('0000', fmt),
        touchratio=0.1,
        ratio_low=0.2,
        ratio_high=0.3,
        loc='upper right',
        date_it=date(2020, 3, 1),
    )
    for item in important_cities:
        print(item)
        plot33_shade(data['sim_cum_confirmed_deduction_s1'][str(item)],min_data['sim_cum_confirmed_deduction_s1'][str(item)],max_data['sim_cum_confirmed_deduction_s1'][str(item)],
              data['sim_cum_confirmed_deduction_s2'][str(item)],min_data['sim_cum_confirmed_deduction_s2'][str(item)],max_data['sim_cum_confirmed_deduction_s2'][str(item)],
              data['sim_cum_confirmed_deduction_s3'][str(item)],min_data['sim_cum_confirmed_deduction_s3'][str(item)],max_data['sim_cum_confirmed_deduction_s3'][str(item)],
              data['sim_cum_confirmed_deduction_s8'][str(item)],min_data['sim_cum_confirmed_deduction_s8'][str(item)],max_data['sim_cum_confirmed_deduction_s8'][str(item)],
              data['real_cum_confirmed'][str(item)],
              code_dict[int(item)],
              'img/{}/cum_confirmed_prediction.{}'.format(str(item), fmt),
              touchratio=data['x'][str(item)][1],
              ratio_low=data['touch_ratio_low'][str(item)],
              ratio_high=data['touch_ratio_hight'][str(item)],
              loc='lower right',
              date_it=date(2020,1,23),
               ext_flag=True
               )
        plot33_shade(data['sim_confirmed_deduction_s1'][str(item)],min_data['sim_confirmed_deduction_s1'][str(item)],max_data['sim_confirmed_deduction_s1'][str(item)],
              data['sim_confirmed_deduction_s2'][str(item)],min_data['sim_confirmed_deduction_s2'][str(item)],max_data['sim_confirmed_deduction_s2'][str(item)],
              data['sim_confirmed_deduction_s3'][str(item)],min_data['sim_confirmed_deduction_s3'][str(item)],max_data['sim_confirmed_deduction_s3'][str(item)],
              data['sim_confirmed_deduction_s8'][str(item)],min_data['sim_confirmed_deduction_s8'][str(item)],max_data['sim_confirmed_deduction_s8'][str(item)],
              data['real_confirmed'][str(item)],
              code_dict[int(item)],
              'img/{}/confirmed_prediction.{}'.format(str(item), fmt),
              touchratio=data['x'][str(item)][1],
              ratio_low=data['touch_ratio_low'][str(item)],
              ratio_high=data['touch_ratio_hight'][str(item)],
              loc='upper right',
              date_it=date(2020,1,23),
               ext_flag=True
               )
        plot33_shade(data['sim_cum_confirmed_deduction_s1'][str(item)],
                     min_data['sim_cum_confirmed_deduction_s1'][str(item)],
                     max_data['sim_cum_confirmed_deduction_s1'][str(item)],
                     data['sim_cum_confirmed_deduction_s5'][str(item)],
                     min_data['sim_cum_confirmed_deduction_s5'][str(item)],
                     max_data['sim_cum_confirmed_deduction_s5'][str(item)],
                     data['sim_cum_confirmed_deduction_s4'][str(item)],
                     min_data['sim_cum_confirmed_deduction_s4'][str(item)],
                     max_data['sim_cum_confirmed_deduction_s4'][str(item)],
                     data['sim_cum_confirmed_deduction_s6'][str(item)],
                     min_data['sim_cum_confirmed_deduction_s6'][str(item)],
                     max_data['sim_cum_confirmed_deduction_s6'][str(item)],
                     data['real_cum_confirmed'][str(item)],
                     code_dict[int(item)],
                     'img/{}/cum_confirmed_prediction_simdate.{}'.format(str(item), fmt),
                     touchratio=data['x'][str(item)][1],
                     ratio_low=data['touch_ratio_low'][str(item)],
                     ratio_high=data['touch_ratio_hight'][str(item)],
                     loc='lower right',
                     date_it=date(2020, 1, 23),
                     #ext_flag=True
                     )
        plot33_shade(data['sim_confirmed_deduction_s1'][str(item)], min_data['sim_confirmed_deduction_s1'][str(item)],
                     max_data['sim_confirmed_deduction_s1'][str(item)],
                     data['sim_confirmed_deduction_s5'][str(item)], min_data['sim_confirmed_deduction_s5'][str(item)],
                     max_data['sim_confirmed_deduction_s5'][str(item)],
                     data['sim_confirmed_deduction_s4'][str(item)], min_data['sim_confirmed_deduction_s4'][str(item)],
                     max_data['sim_confirmed_deduction_s4'][str(item)],
                     data['sim_confirmed_deduction_s6'][str(item)], min_data['sim_confirmed_deduction_s4'][str(item)],
                     max_data['sim_confirmed_deduction_s6'][str(item)],
                     data['real_confirmed'][str(item)],
                     code_dict[int(item)],
                     'img/{}/confirmed_prediction_simdate.{}'.format(str(item), fmt),
                     touchratio=data['x'][str(item)][1],
                     ratio_low=data['touch_ratio_low'][str(item)],
                     ratio_high=data['touch_ratio_hight'][str(item)],
                     loc='upper right',
                     date_it=date(2020, 1, 23),
                     #ext_flag=True
                     )
        plot33_time_shade(
                    data['sim_confirmed_deduction_s1'][str(item)],min_data['sim_confirmed_deduction_s1'][str(item)],max_data['sim_confirmed_deduction_s1'][str(item)],
                    data['sim_confirmed_deduction_s9'][str(item)],min_data['sim_confirmed_deduction_s9'][str(item)],max_data['sim_confirmed_deduction_s9'][str(item)],
                    data['sim_confirmed_deduction_s10'][str(item)],min_data['sim_confirmed_deduction_s10'][str(item)],max_data['sim_confirmed_deduction_s10'][str(item)],
                    data['sim_confirmed_deduction_s11'][str(item)],min_data['sim_confirmed_deduction_s11'][str(item)],max_data['sim_confirmed_deduction_s11'][str(item)],
                    data['sim_confirmed_deduction_s12'][str(item)],min_data['sim_confirmed_deduction_s12'][str(item)],max_data['sim_confirmed_deduction_s12'][str(item)],
            data['real_confirmed'][str(item)],
            code_dict[item],
                     'img/{}/confirmed_prediction_simdate_diff.{}'.format(str(item), fmt),
                     touchratio=1,
                     ratio_high=1.5,
                     ratio_low=0.5,
                     loc='upper right')
        plot33_time_shade(
            data['sim_cum_confirmed_deduction_s1'][str(item)], min_data['sim_cum_confirmed_deduction_s1'][str(item)],
            max_data['sim_cum_confirmed_deduction_s1'][str(item)],
            data['sim_cum_confirmed_deduction_s9'][str(item)], min_data['sim_cum_confirmed_deduction_s9'][str(item)],
            max_data['sim_cum_confirmed_deduction_s9'][str(item)],
            data['sim_cum_confirmed_deduction_s10'][str(item)], min_data['sim_cum_confirmed_deduction_s10'][str(item)],
            max_data['sim_cum_confirmed_deduction_s10'][str(item)],
            data['sim_cum_confirmed_deduction_s11'][str(item)], min_data['sim_cum_confirmed_deduction_s11'][str(item)],
            max_data['sim_cum_confirmed_deduction_s11'][str(item)],
            data['sim_cum_confirmed_deduction_s12'][str(item)], min_data['sim_cum_confirmed_deduction_s12'][str(item)],
            max_data['sim_cum_confirmed_deduction_s12'][str(item)],
            data['real_cum_confirmed'][str(item)],
            code_dict[item],
            'img/{}/cum_confirmed_prediction_simdate_diff.{}'.format(str(item), fmt),
            touchratio=1,
            ratio_high=1.5,
            ratio_low=0.5,
            loc='upper right')


    print('==================================', str(final_date))
    print('March 1 different control measure: ')
    print('confirmed')
    print('middle: ')
    print('100\%', format(int(get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities)[-1]), ','),
          '190\%', format(int(get_total(data, 'sim_cum_confirmed_deduction_s5', total_cities)[-1]), ','),
          '160\%', format(int(get_total(data, 'sim_cum_confirmed_deduction_s4', total_cities)[-1]), ','),
          '210\%', format(int(get_total(data, 'sim_cum_confirmed_deduction_s6', total_cities)[-1]), ','), )
    print('min: ')
    print('100\%', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s1', total_cities)[-1]), ','),
          '190\%', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s5', total_cities)[-1]), ','),
          '160\%', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s4', total_cities)[-1]), ','),
          '210\%', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s6', total_cities)[-1]), ','), )
    print('max:')
    print('100\%', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s1', total_cities)[-1]), ','),
          '190\%', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s5', total_cities)[-1]), ','),
          '160\%', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s4', total_cities)[-1]), ','),
          '210\%', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s6', total_cities)[-1]), ','), )
    print('infected')
    print('middle: ')
    print('100\%', format(int(get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[-1]), ','),
          '190\%', format(int(get_total(data, 'sim_cum_infection_deduction_s5', total_cities)[-1]), ','),
          '160\%', format(int(get_total(data, 'sim_cum_infection_deduction_s4', total_cities)[-1]), ','),
          '210\%', format(int(get_total(data, 'sim_cum_infection_deduction_s6', total_cities)[-1]), ','), )
    print('min: ')
    print('100\%', format(int(get_total(min_data, 'sim_cum_infection_deduction_s1', total_cities)[-1]), ','),
          '190\%', format(int(get_total(min_data, 'sim_cum_infection_deduction_s5', total_cities)[-1]), ','),
          '160\%', format(int(get_total(min_data, 'sim_cum_infection_deduction_s4', total_cities)[-1]), ','),
          '210\%', format(int(get_total(min_data, 'sim_cum_infection_deduction_s6', total_cities)[-1]), ','), )
    print('max:')
    print('100\%', format(int(get_total(max_data, 'sim_cum_infection_deduction_s1', total_cities)[-1]), ','),
          '190\%', format(int(get_total(max_data, 'sim_cum_infection_deduction_s5', total_cities)[-1]), ','),
          '160\%', format(int(get_total(max_data, 'sim_cum_infection_deduction_s4', total_cities)[-1]), ','),
          '210\%', format(int(get_total(max_data, 'sim_cum_infection_deduction_s6', total_cities)[-1]), ','), )
    print('==================================', str(final_date))

    plot1_shade(get_total(data, 'real_cum_confirmed', total_cities),
                get_total(data, 'sim_cum_confirmed', total_cities),
                get_total(min_data, 'sim_cum_confirmed', total_cities),
                get_total(max_data, 'sim_cum_confirmed', total_cities),
                'China\'s mainland',
                'img/{}/cum_real_sim.{}'.format('0000', fmt))
    plot1_shade(get_total(data, 'real_confirmed', total_cities),
                get_total(data, 'sim_confirmed', total_cities),
                get_total(min_data, 'sim_confirmed', total_cities),
                get_total(max_data, 'sim_confirmed', total_cities),
                'China\'s mainland',
                'img/{}/increase_real_sim.{}'.format('0000', fmt))
    plot1_shade(get_total(data, 'real_cum_dead', total_cities),
                get_total(data, 'sim_cum_dead_s1', total_cities),
                get_total(min_data, 'sim_cum_dead_s1', total_cities),
                get_total(max_data, 'sim_cum_dead_s1', total_cities),
                'China\'s mainland',
                'img/{}/cum_real_sim_dead.{}'.format('0000', fmt),
                interval=20)
    plot33_time_shade(
        get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities),
        get_total(min_data, 'sim_cum_confirmed_deduction_s1', total_cities),
        get_total(max_data, 'sim_cum_confirmed_deduction_s1', total_cities),
        get_total(data, 'sim_cum_confirmed_deduction_s9', total_cities),
        get_total(min_data, 'sim_cum_confirmed_deduction_s9', total_cities),
        get_total(max_data, 'sim_cum_confirmed_deduction_s9', total_cities),
        get_total(data, 'sim_cum_confirmed_deduction_s10', total_cities),
        get_total(min_data, 'sim_cum_confirmed_deduction_s10', total_cities),
        get_total(max_data, 'sim_cum_confirmed_deduction_s10', total_cities),
        get_total(data, 'sim_cum_confirmed_deduction_s11', total_cities),
        get_total(min_data, 'sim_cum_confirmed_deduction_s11', total_cities),
        get_total(max_data, 'sim_cum_confirmed_deduction_s11', total_cities),
        get_total(data, 'sim_cum_confirmed_deduction_s12', total_cities),
        get_total(min_data, 'sim_cum_confirmed_deduction_s12', total_cities),
        get_total(max_data, 'sim_cum_confirmed_deduction_s12', total_cities),
        get_total(data, 'real_cum_confirmed', total_cities),
        'China\'s mainland',
        'img/{}/cum_confirmed_prediction_time.{}'.format('0000', fmt),
        touchratio=1,
        ratio_high=1.5,
        ratio_low=0.5,
        loc='upper right'
    )
    plot33_time_shade(
        get_total(data, 'sim_confirmed_deduction_s1', total_cities),
        get_total(min_data, 'sim_confirmed_deduction_s1', total_cities),
        get_total(max_data, 'sim_confirmed_deduction_s1', total_cities),
        get_total(data, 'sim_confirmed_deduction_s9', total_cities),
        get_total(min_data, 'sim_confirmed_deduction_s9', total_cities),
        get_total(max_data, 'sim_confirmed_deduction_s9', total_cities),
        get_total(data, 'sim_confirmed_deduction_s10', total_cities),
        get_total(min_data, 'sim_confirmed_deduction_s10', total_cities),
        get_total(max_data, 'sim_confirmed_deduction_s10', total_cities),
        get_total(data, 'sim_confirmed_deduction_s11', total_cities),
        get_total(min_data, 'sim_confirmed_deduction_s11', total_cities),
        get_total(max_data, 'sim_confirmed_deduction_s11', total_cities),
        get_total(data, 'sim_confirmed_deduction_s12', total_cities),
        get_total(min_data, 'sim_confirmed_deduction_s12', total_cities),
        get_total(max_data, 'sim_confirmed_deduction_s12', total_cities),
        get_total(data, 'real_confirmed', total_cities),
        'China\'s mainland',
        'img/{}/confirmed_prediction_time.{}'.format('0000', fmt),
        touchratio=1,
        ratio_high=1.5,
        ratio_low=0.5,
        loc='upper right'
    )
    print('============== parameter')
    for item in total_cities:
        print(code_dict[item])
        print('RMSE: {} ({}-{})'.format(data['newly_confirmed_loss'][str(item)],
                                        min_data['newly_confirmed_loss'][str(item)],
                                        max_data['newly_confirmed_loss'][str(item)]))
        print('RO1: {} ({}-{})'.format(data['R01'][str(item)],
                                        min_data['R01'][str(item)],
                                        max_data['R01'][str(item)]))
        print('RO2: {} ({}-{})'.format(data['R02'][str(item)],
                                       min_data['R02'][str(item)],
                                       max_data['R02'][str(item)]))
        print('DT1: {} ({}-{})'.format(data['DT1'][str(item)],
                                       min_data['DT1'][str(item)],
                                       max_data['DT1'][str(item)]))
        print('DT2: {} ({}-{})'.format(data['DT2'][str(item)],
                                       min_data['DT2'][str(item)],
                                       max_data['DT2'][str(item)]))
        print('Death ratio: {} ({}-{})'.format(data['death_rate'][str(item)],
                                       min_data['death_rate'][str(item)],
                                       max_data['death_rate'][str(item)]))
    print('============== parameter')
    print('==================================', str(final_date))
    print('Jan 23 different control date: ')
    print('confirmed')
    print('middle: ')
    print('Jan 23', format(int(get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities)[-1]), ','),
          'Jan 24', format(int(get_total(data, 'sim_cum_confirmed_deduction_s9', total_cities)[-1]), ','),
          'Jan 26', format(int(get_total(data, 'sim_cum_confirmed_deduction_s10', total_cities)[-1]), ','),
          'Jan 28', format(int(get_total(data, 'sim_cum_confirmed_deduction_s11', total_cities)[-1]), ','),
          'Jan 30', format(int(get_total(data, 'sim_cum_confirmed_deduction_s12', total_cities)[-1]), ','), )
    print('min: ')
    print('Jan 23', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s1', total_cities)[-1]), ','),
          'Jan 24', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s9', total_cities)[-1]), ','),
          'Jan 26', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s10', total_cities)[-1]), ','),
          'Jan 28', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s11', total_cities)[-1]), ','),
          'Jan 30', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s12', total_cities)[-1]), ','), )
    print('max:')
    print('Jan 23', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s1', total_cities)[-1]), ','),
          'Jan 24', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s9', total_cities)[-1]), ','),
          'Jan 26', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s10', total_cities)[-1]), ','),
          'Jan 28', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s11', total_cities)[-1]), ','),
          'Jan 30', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s12', total_cities)[-1]), ','), )
    print('infected')
    print('middle: ')
    print('Jan 23', format(int(get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[-1]), ','),
          'Jan 24', format(int(get_total(data, 'sim_cum_infection_deduction_s9', total_cities)[-1]), ','),
          'Jan 26', format(int(get_total(data, 'sim_cum_infection_deduction_s10', total_cities)[-1]), ','),
          'Jan 28', format(int(get_total(data, 'sim_cum_infection_deduction_s11', total_cities)[-1]), ','),
          'Jan 30', format(int(get_total(data, 'sim_cum_infection_deduction_s12', total_cities)[-1]), ','), )
    print('min: ')
    print('Jan 23', format(int(get_total(min_data, 'sim_cum_infection_deduction_s1', total_cities)[-1]), ','),
          'Jan 24', format(int(get_total(min_data, 'sim_cum_infection_deduction_s9', total_cities)[-1]), ','),
          'Jan 26', format(int(get_total(min_data, 'sim_cum_infection_deduction_s10', total_cities)[-1]), ','),
          'Jan 28', format(int(get_total(min_data, 'sim_cum_infection_deduction_s11', total_cities)[-1]), ','),
          'Jan 30', format(int(get_total(min_data, 'sim_cum_infection_deduction_s12', total_cities)[-1]), ','), )
    print('max:')
    print('Jan 23', format(int(get_total(max_data, 'sim_cum_infection_deduction_s1', total_cities)[-1]), ','),
          'Jan 24', format(int(get_total(max_data, 'sim_cum_infection_deduction_s9', total_cities)[-1]), ','),
          'Jan 26', format(int(get_total(max_data, 'sim_cum_infection_deduction_s10', total_cities)[-1]), ','),
          'Jan 28', format(int(get_total(max_data, 'sim_cum_infection_deduction_s11', total_cities)[-1]), ','),
          'Jan 30', format(int(get_total(max_data, 'sim_cum_infection_deduction_s12', total_cities)[-1]), ','), )
    print('==================================', str(final_date))

    plot12_shade(get_total(data, 'real_confirmed', total_cities),
           get_total(data, 'sim_confirmed', total_cities), get_total(min_data, 'sim_confirmed', total_cities), get_total(max_data, 'sim_confirmed', total_cities),
           get_total(data, 'sim_new_infection', total_cities),  get_total(min_data, 'sim_new_infection', total_cities), get_total(max_data, 'sim_new_infection', total_cities),
           'The newly number of confirmed cases ',
           'img/{}/increase_real_sim_infect.{}'.format('0000', fmt))
    plot12_shade(get_total(data, 'real_cum_confirmed', total_cities),
           get_total(data, 'sim_cum_confirmed', total_cities), get_total(min_data, 'sim_cum_confirmed', total_cities), get_total(max_data, 'sim_cum_confirmed', total_cities),
           get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[:lreal], get_total(min_data, 'sim_cum_infection_deduction_s1', total_cities)[:lreal],get_total(max_data, 'sim_cum_infection_deduction_s1', total_cities)[:lreal],
           'The newly number of confirmed cases ',
           'img/{}/cum_real_sim_infect.{}'.format('0000', fmt))

    bias = (today - date(2020,3,13)).days - 1
    print('==================================', str(cur_date + timedelta(bias)))
    print('current total infections and confirmed cases')
    print('confirmed')
    print('middle: ')
    print('100\%', format(int(get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities)[lreal-1 + bias]), ','))
    print('min: ')
    print('100\%', format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s1', total_cities)[lreal-1  + bias]), ','))
    print('max: ')
    print('100\%', format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s1', total_cities)[lreal-1+ bias]), ','))
    print('real', format(int(get_total(data,'real_cum_confirmed', total_cities)[-1]), ','))
    print('infected')
    print('middle: ')
    print('100\%', format(int(get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[lreal-1+ bias]), ','))
    print('min: ')
    print('100\%', format(int(get_total(min_data, 'sim_cum_infection_deduction_s1', total_cities)[lreal-1+ bias]), ','))
    print('max: ')
    print('100\%', format(int(get_total(max_data, 'sim_cum_infection_deduction_s1', total_cities)[lreal-1+ bias]), ','))
    print('infe-confi: ')
    print('{} ({}-{})'.format(format(int(data['sim_cum_infe_minus_conf_s1']['0000'][lreal-1+bias]),','),
                              format(int(min_data['sim_cum_infe_minus_conf_s1']['0000'][lreal - 1 + bias]), ','),
                              format(int(max_data['sim_cum_infe_minus_conf_s1']['0000'][lreal - 1 + bias]), ',')))
    print('ratio')
    print('middle: ')
    print('100\%', format(int(get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[lreal - 1+ bias]), ','))
    print('min: ')
    print('100\%', format(int(get_total(min_data, 'sim_cum_infection_deduction_s1', total_cities)[lreal - 1+ bias]), ','))
    print('max: ')
    print('100\%', format(int(get_total(max_data, 'sim_cum_infection_deduction_s1', total_cities)[lreal - 1+ bias]), ','))
    print('cum self_cure')
    print('100\%', format(int(get_total(data, 'sim_cum_self_cured_deduction_s1', total_cities)[lreal - 1+ bias]), ','),
          '({}-{})'.format(format(int(get_total(min_data,'sim_cum_self_cured_deduction_s1', total_cities)[lreal - 1+ bias]), ','),
                           format(int(get_total(max_data,'sim_cum_self_cured_deduction_s1', total_cities)[lreal - 1+ bias]), ',')))
    print('total infection')
    print('100\%', format(int(get_total(data, 'sim_total_infection_deduction_s1', total_cities)[lreal - 1+ bias]), ','),
          '({}-{})'.format(format(int(get_total(min_data,'sim_total_infection_deduction_s1', total_cities)[lreal - 1+ bias]), ','),
                           format(int(get_total(max_data,'sim_total_infection_deduction_s1', total_cities)[lreal - 1+ bias]), ',')))
    print('nosymbol')
    print('100\%', format(int(get_total(data, 'sim_cum_nosymbol_deduction_s1', total_cities)[lreal - 1+ bias]), ','),
          '({}-{})'.format(format(int(get_total(min_data,'sim_cum_nosymbol_deduction_s1', total_cities)[lreal - 1+ bias]), ','),
                           format(int(get_total(max_data,'sim_cum_nosymbol_deduction_s1', total_cities)[lreal - 1+ bias]), ',')))
    print('total_iso')
    # sim_total_isolation_deduction_s1
    print('100\%', format(int(get_total(data, 'sim_total_isolation_deduction_s1', total_cities)[lreal - 1+ bias]), ','),
          '({}-{})'.format(format(int(get_total(min_data,'sim_total_isolation_deduction_s1', total_cities)[lreal - 1+ bias]), ','),
                           format(int(get_total(max_data,'sim_total_isolation_deduction_s1', total_cities)[lreal - 1+ bias]), ',')))
    print('ratio cur')
    print('{} ({}-{})'.format(data['current_asym']['0000'], min_data['current_asym']['0000'], max_data['current_asym']['0000']))
    print('final')
    print('{} ({}-{})'.format(data['final_asym']['0000'], min_data['final_asym']['0000'], max_data['final_asym']['0000']))

    print('==================================', str(cur_date))
    print('\n\n')
    cur_data_4_03 = json.load(open('./data/cur_confirmed-{}.json'.format(str(today)), 'r'))
    for item in total_cities:
        print('{} &'.format(code_dict[item]), sep='', end='')
        print('{} &'.format(format(int(cur_data_4_03[str(item)]), ',')), sep='',end='')
        print('{} &'.format(format(int(data['sim_cum_confirmed_deduction_s1'][str(item)][lreal-1+bias]), ',')), sep='',end='')
        print('{} &'.format(format(int(data['sim_cum_infection_deduction_s1'][str(item)][lreal-1+bias]), ',')), sep='',end='')
        print('{} &'.format(format(int(data['sim_total_infection_deduction_s1'][str(item)][lreal-1+bias]), ',')), sep='',end='')
        print('{} \\\\'.format(format(int(data['sim_cum_self_cured_deduction_s1'][str(item)][lreal-1+bias]), ',')), sep='',end='')
        print('')



        pass
    print('\n\n')
    print('{} &'.format('China\'s Mainland'), sep='', end='')
    total_it = sum([cur_data_4_03[key] for key in cur_data_4_03.keys()])
    print('{} &'.format(format(int(total_it), ',')), sep='', end='')
    print('{} ({}-{}) &'.format(
        format(int(get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities)[lreal - 1 + bias]), ','),
        format(int(get_total(min_data, 'sim_cum_confirmed_deduction_s1', total_cities)[lreal - 1 + bias]), ','),
        format(int(get_total(max_data, 'sim_cum_confirmed_deduction_s1', total_cities)[lreal - 1 + bias]), ','),
    ), sep='', end='')

    print('{} ({}-{}) &'.format(
        format(int(get_total(data, 'sim_cum_infection_deduction_s1', total_cities)[lreal - 1 + bias]), ','),
        format(int(get_total(min_data, 'sim_cum_infection_deduction_s1', total_cities)[lreal - 1 + bias]), ','),
        format(int(get_total(max_data, 'sim_cum_infection_deduction_s1', total_cities)[lreal - 1 + bias]), ','),
    ), sep='', end='')

    print('{} ({}-{}) &'.format(
        format(int(get_total(data, 'sim_total_infection_deduction_s1', total_cities)[lreal - 1 + bias]), ','),
        format(int(get_total(min_data, 'sim_total_infection_deduction_s1', total_cities)[lreal - 1 + bias]), ','),
        format(int(get_total(max_data, 'sim_total_infection_deduction_s1', total_cities)[lreal - 1 + bias]), ','),
    ), sep='', end='')

    print('{} ({}-{}) \\\\'.format(
        format(int(get_total(data, 'sim_cum_self_cured_deduction_s1', total_cities)[lreal - 1 + bias]), ','),
        format(int(get_total(min_data, 'sim_cum_self_cured_deduction_s1', total_cities)[lreal - 1 + bias]), ','),
        format(int(get_total(max_data, 'sim_cum_self_cured_deduction_s1', total_cities)[lreal - 1 + bias]), ','),
    ), sep='', end='')
    print('')
    for item in total_cities:
        print('{} &'.format(code_dict[item]), sep='', end='')
        print('{} &'.format(format(int(cur_data_4_03[str(item)]), ',')), sep='',end='')
        print('{} ({}-{}) &'.format(
            format(int(data['sim_cum_confirmed_deduction_s1'][str(item)][lreal-1+bias]), ','),
            format(int(min_data['sim_cum_confirmed_deduction_s1'][str(item)][lreal - 1 + bias]), ','),
            format(int(max_data['sim_cum_confirmed_deduction_s1'][str(item)][lreal - 1 + bias]), ','),
        ), sep='',end='')
        print('{} ({}-{}) &'.format(
            format(int(data['sim_cum_infection_deduction_s1'][str(item)][lreal-1+bias]), ','),
            format(int(min_data['sim_cum_infection_deduction_s1'][str(item)][lreal - 1 + bias]), ','),
            format(int(max_data['sim_cum_infection_deduction_s1'][str(item)][lreal - 1 + bias]), ','),
        ), sep='',end='')
        print('{} ({}-{}) &'.format(
            format(int(data['sim_total_infection_deduction_s1'][str(item)][lreal-1+bias]), ','),
            format(int(min_data['sim_total_infection_deduction_s1'][str(item)][lreal - 1 + bias]), ','),
            format(int(max_data['sim_total_infection_deduction_s1'][str(item)][lreal - 1 + bias]), ','),
        ), sep='',end='')
        print('{} ({}-{}) \\\\'.format(
            format(int(data['sim_cum_self_cured_deduction_s1'][str(item)][lreal-1+bias]), ','),
            format(int(min_data['sim_cum_self_cured_deduction_s1'][str(item)][lreal - 1 + bias]), ','),
            format(int(max_data['sim_cum_self_cured_deduction_s1'][str(item)][lreal - 1 + bias]), ','),
        ), sep='',end='')
        print('')

    # print variables
    x_list = construct_x(data, total_cities)
    min_x_list = construct_x(min_data, total_cities)
    max_x_list = construct_x(max_data, total_cities)
    format_out(x_list, min_x_list, max_x_list)

    exit(0)

if __name__ == '__main__':
    main()
