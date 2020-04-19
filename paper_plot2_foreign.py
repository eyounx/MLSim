from show_simulation_foreign import plot1, plot3, plot1_shade, plot3_shade, plot12_shade
from components.utils import get_data_path, get_important_date, get_seed_num, it_code_dict, format_out, construct_x
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

def main():
    data = {}
    code_dict = it_code_dict()
    fmt = 'pdf'

    #total_cities = pd.read_csv("./data/province_data.csv")['adcode'].unique()
    total_cities = list(pd.read_csv(get_data_path(True))['adcode'].unique())


    important_cities = total_cities
    important_cities = [900003, 900004, 900005, 900006, 900007, 900008, 900009, 900010]
    with open('./data_run_foreign_middle.json', 'r') as f:
        data = json.load(f)
    with open('./data_run_foreign_xmin.json', 'r') as f:
        min_data = json.load(f)
    with open('./data_run_foreign_xmax.json', 'r') as f:
        max_data = json.load(f)
    if not os.path.exists('./img'):
        os.mkdir('./img')
    if not os.path.exists('./img/0000'):
        os.makedirs('./img/0000')
    for item in important_cities:
        it_path = './img/{}'.format(str(item))
        if not os.path.exists(it_path):
            os.mkdir(it_path)
    print("final day: ", date(2020,1,24) + timedelta(days=len(get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities))))
    final_date = date(2020, 1, 24) + timedelta(days=len(get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities)))
    cur_date = date(2020,1, 24) + timedelta(days=len(get_total(data, 'real_confirmed', total_cities)))
    print('early')
    for item in important_cities:
        print(code_dict[int(item)],": ",data['sim_cum_confirmed_deduction_s3'][str(item)][-1])
        print(len(data['real_confirmed'][str(item)]))
    print('late')
    for item in important_cities:
        print(code_dict[int(item)],": ",data['sim_cum_confirmed_deduction_s4'][str(item)][-1])
    print('3-20')
    for item in important_cities:
        print(code_dict[int(item)],": ",data['sim_cum_confirmed_deduction_s6'][str(item)][-1])
    city_d = [900004, 900005, 900006, 900007, 900008, 900003, 900009, 900010]

    print('\n\n===========')
    print('country ', sep='', end='')
    for item in city_d:
        print('& ', code_dict[item], sep='', end='')
    print('\\\\')
    print('total confirmed cases', sep='', end='')
    for item in city_d:
        print(' & ', int(data['sim_cum_confirmed_deduction_s1'][str(item)][-1]), sep='', end='')

    print('\\\\')
    print('total infections' , sep='', end='')
    for item in city_d:
        print(' & ', int(data['sim_cum_infection_deduction_s1'][str(item)][-1]), sep='', end='')

    print('\\\\')
    print('===========\n\n')

    print('Date when $k$ was changed & ', sep='', end='')
    for item in city_d:
        print(code_dict[item], ' & ', sep='', end='')
    print('\\\\')
    print('middle &', sep='', end='')
    for item in city_d:
        print(int(data['sim_cum_confirmed_deduction_s1'][str(item)][-1]), ' & ', sep='',end='')
        print(int(data['sim_cum_infection_deduction_s1'][str(item)][-1]), ' & ', sep='', end='')
    print('\\\\')
    print('min&', sep='', end='')
    for item in city_d:
        print(int(min_data['sim_cum_confirmed_deduction_s1'][str(item)][-1]), ' & ', sep='',end='')
        print(int(min_data['sim_cum_infection_deduction_s1'][str(item)][-1]), ' & ', sep='', end='')
    print('\\\\')
    print('max&', sep='', end='')
    for item in city_d:
        print(int(max_data['sim_cum_confirmed_deduction_s1'][str(item)][-1]), ' & ', sep='',end='')
        print(int(max_data['sim_cum_infection_deduction_s1'][str(item)][-1]), ' & ', sep='', end='')
    print('\\\\')
    print('country&real confirmed&MLSim confirmed&MLSim infected& proportion of asymptomatic\\\\\\hline')
    lreal = len(get_total(data, 'real_cum_confirmed', total_cities))
    for item in city_d:
        real_cum = data['real_cum_confirmed'][str(item)][-1]
        sim_cum_confirm = data['sim_cum_confirmed'][str(item)][-1]
        sim_cum_infect = data['sim_cum_infection_deduction_s1'][str(item)][lreal - 1]

        print(code_dict[int(item)], ' & ',
              real_cum, ' & ',
              int(sim_cum_confirm), ' & ',
              int(sim_cum_infect), ' & ',
              np.round((1 - sim_cum_confirm / sim_cum_infect) * 100, 3), '$\\%$\\\\ ', sep='')
    #exit(0)
    lreal = len(get_total(data, 'real_cum_confirmed', total_cities))
    # draw compare to real data for every city and full
    print('drawing real/sim compare')
    for item in important_cities:
        print(item)
        plot1_shade(data['real_cum_confirmed'][str(item)],
                    data['sim_cum_confirmed'][str(item)],
                    min_data['sim_cum_confirmed'][str(item)],
                    max_data['sim_cum_confirmed'][str(item)],
                    'The cumulative number of confirmed cases ',
              'img/{}/cum_real_sim.{}'.format(str(item), fmt))
        plot1_shade(data['real_confirmed'][str(item)],
              data['sim_confirmed'][str(item)],
              min_data['sim_confirmed'][str(item)],
              max_data['sim_confirmed'][str(item)],
              'The newly number of confirmed cases ',
          'img/{}/increase_real_sim.{}'.format(str(item), fmt))
    print('drawing real/sim compare forecast')
    for item in important_cities:
        print(item)
        plot1_shade(data['real_cum_confirmed'][str(item)],
              data['sim_cum_confirmed_deduction_s1'][str(item)],
              min_data['sim_cum_confirmed_deduction_s1'][str(item)],
              max_data['sim_cum_confirmed_deduction_s1'][str(item)],
              'The cumulative number of confirmed cases ',
              'img/{}/only_cum_real_sim.{}'.format(str(item), fmt), 18,
                    date_it=get_important_date(item))
        plot1_shade(data['real_confirmed'][str(item)],
                        data['sim_confirmed_deduction_s1'][str(item)],
                        min_data['sim_confirmed_deduction_s1'][str(item)],
                        max_data['sim_confirmed_deduction_s1'][str(item)],
                        'The newly number of confirmed cases ',
                        'img/{}/only_increase_real_sim.{}'.format(str(item), fmt), 18, date_it=get_important_date(item))
        plot12_shade(data['real_confirmed'][str(item)],
                     data['sim_confirmed'][str(item)], min_data['sim_confirmed'][str(item)],  max_data['sim_confirmed'][str(item)],
                     data['sim_new_infection'][str(item)], min_data['sim_new_infection'][str(item)], max_data['sim_new_infection'][str(item)],
           'The newly number of confirmed cases ',
           'img/{}/increase_real_sim_infect.{}'.format(str(item), fmt))
        plot12_shade(data['real_cum_confirmed'][str(item)],
                     data['sim_cum_confirmed'][str(item)], min_data['sim_confirmed'][str(item)],
                     max_data['sim_confirmed'][str(item)],
                     data['sim_cum_infection_deduction_s1'][str(item)][:lreal], min_data['sim_cum_infection_deduction_s1'][str(item)][:lreal],
                     max_data['sim_cum_infection_deduction_s1'][str(item)][:lreal],
                     'The newly number of confirmed cases ',
                     'img/{}/cum_real_sim_infect.{}'.format(str(item), fmt))
        print('==================================', code_dict[item], str(cur_date))
        print('current total infections and confirmed cases')
        print('confirmed')
        print('middle: ')
        print('100\%', format(int(data['sim_cum_confirmed'][str(item)][lreal-1]), ','),
                              '({}-{})'.format(format(int(min_data['sim_cum_confirmed'][str(item)][lreal-1]), ','),
                                               format(int(max_data['sim_cum_confirmed'][str(item)][lreal-1]), ',')))
        print('min: ')
        print('100\%', format(int(min_data['sim_cum_confirmed'][str(item)][lreal-1]), ','))
        print('max: ')
        print('100\%', format(int(max_data['sim_cum_confirmed'][str(item)][lreal-1]), ','))
        print('real: ', format(int(data['real_cum_confirmed'][str(item)][-1]), ','))
        print('infected')
        print('middle: ')
        print('100\%', format(int(data['sim_cum_infection_deduction_s1'][str(item)][lreal-1]), ','),
                      '({}-{})'.format(format(int(min_data['sim_cum_infection_deduction_s1'][str(item)][lreal-1]), ','),
                                       format(int(max_data['sim_cum_infection_deduction_s1'][str(item)][lreal-1]), ',')))
        print('min: ')
        print('100\%', format(int(min_data['sim_cum_infection_deduction_s1'][str(item)][lreal-1]), ','))
        print('max: ')
        print('100\%', format(int(max_data['sim_cum_infection_deduction_s1'][str(item)][lreal-1]), ','))
        print('asym')
        print('middle: ')
        print('100\%', format(int(data['current_asym'][str(item)]), ','))
        print('min: ')
        print('100\%', format(int(min_data['current_asym'][str(item)]), ','))
        print('max: ')
        print('100\%', format(int(max_data['current_asym'][str(item)]), ','))
        print('final asym')
        print('middle: ')
        print('100\%', format(int(data['final_asym'][str(item)]), ','))
        print('min: ')
        print('100\%', format(int(min_data['final_asym'][str(item)]), ','))
        print('max: ')
        print('100\%', format(int(max_data['final_asym'][str(item)]), ','))
        print('cum self_cure')
        print('100\%', format(int(data['sim_cum_self_cured_deduction_s1'][str(item)][lreal - 1]), ','),
              '({}-{})'.format(format(int(min_data['sim_cum_self_cured_deduction_s1'][str(item)][lreal - 1]), ','),
                               format(int(max_data['sim_cum_self_cured_deduction_s1'][str(item)][lreal - 1]), ',')))
        print('total infection')
        print('100\%', format(int(data['sim_total_infection_deduction_s1'][str(item)][lreal - 1]), ','),
              '({}-{})'.format(format(int(min_data['sim_total_infection_deduction_s1'][str(item)][lreal - 1]), ','),
                               format(int(max_data['sim_total_infection_deduction_s1'][str(item)][lreal - 1]), ',')))
        print('nosymbol')
        print('100\%', format(int(data['sim_cum_nosymbol_deduction_s1'][str(item)][lreal - 1]), ','),
              '({}-{})'.format(format(int(min_data['sim_cum_nosymbol_deduction_s1'][str(item)][lreal - 1]), ','),
                               format(int(max_data['sim_cum_nosymbol_deduction_s1'][str(item)][lreal - 1]), ',')))
        # sim_total_isolation_deduction_s1
        print('total iso')
        print('100\%', format(int(data['sim_total_isolation_deduction_s1'][str(item)][lreal - 1]), ','),
              '({}-{})'.format(format(int(min_data['sim_total_isolation_deduction_s1'][str(item)][lreal - 1]), ','),
                               format(int(max_data['sim_total_isolation_deduction_s1'][str(item)][lreal - 1]), ',')))

        print('ratio')
        print('{:.3f} ({:.3f}-{:.3f})'.format(1-data['current_asym'][str(item)],
                                  1-min_data['current_asym'][str(item)],
                                  1-max_data['current_asym'][str(item)]))
        print('==================================', code_dict[item], str(cur_date))
    """
    plot1(get_total(data, 'real_cum_confirmed',total_cities), get_total(data,'sim_cum_confirmed', total_cities),
          'The cumulative number of confirmed cases ',
          'img/{}/cum_real_sim.{}'.format('0000', fmt))
    plot1(get_total(data, 'real_confirmed',total_cities), get_total(data,'sim_confirmed', total_cities),
          'The newly number of confirmed cases ',
          'img/{}/increase_real_sim.{}'.format('0000', fmt))
    """
    # draw different deduction in Feb 24th
    print('drawing different deduction')
    for item in important_cities:
        print(item)
        it_max = None
        if item == 900004:
            it_max = 85000
        if item == 900005:
            it_max = 10000
        if item == 900006:
            it_max = 15000
        if item == 900007:
            it_max = 20000
        if item == 900008:
            it_max = 25000
        if item == 900009:
            it_max = 10000
        if item == 900010:
            it_max = 15000
        it_min = 0
        plot3(data['sim_cum_confirmed_deduction_s1'][str(item)],
              data['sim_cum_confirmed_deduction_s2'][str(item)],
              data['sim_cum_confirmed_deduction_s3'][str(item)],
              data['real_cum_confirmed'][str(item)],
              'Prediction of the cumulative number of confirmed cases',
              'img/{}/cum_confirmed_prediction.{}'.format(str(item), fmt),
              touchratio=data['x'][str(item)][1],
              ratio_low=data['touch_ratio_low'][str(item)],
              ratio_high=data['touch_ratio_hight'][str(item)],
              date_it=get_important_date(item),
              start_date_it=date(2020,2,15))
        plot3(data['sim_confirmed_deduction_s1'][str(item)],
              data['sim_confirmed_deduction_s2'][str(item)],
              data['sim_confirmed_deduction_s3'][str(item)],
              data['real_confirmed'][str(item)],
              'Prediction of the cumulative number of confirmed cases',
              'img/{}/confirmed_prediction.{}'.format(str(item), fmt),
              touchratio=data['x'][str(item)][1],
              ratio_low=data['touch_ratio_low'][str(item)],
              ratio_high=data['touch_ratio_hight'][str(item)],
              date_it=get_important_date(item),
              it_max=it_max,
              it_min=it_min,
              start_date_it=date(2020,2,15))
        plot3_shade(
            data['sim_cum_confirmed_deduction_s1'][str(item)], min_data['sim_cum_confirmed_deduction_s1'][str(item)], max_data['sim_cum_confirmed_deduction_s1'][str(item)],
            data['sim_cum_confirmed_deduction_s2'][str(item)], min_data['sim_cum_confirmed_deduction_s2'][str(item)], max_data['sim_cum_confirmed_deduction_s2'][str(item)],
            data['sim_cum_confirmed_deduction_s3'][str(item)], min_data['sim_cum_confirmed_deduction_s3'][str(item)], max_data['sim_cum_confirmed_deduction_s3'][str(item)],
            data['sim_cum_confirmed_deduction_s8'][str(item)], min_data['sim_cum_confirmed_deduction_s8'][str(item)],
            max_data['sim_cum_confirmed_deduction_s8'][str(item)],
            data['real_cum_confirmed'][str(item)],
              'Prediction of the cumulative number of confirmed cases',
              'img/{}/cum_confirmed_prediction_shade.{}'.format(str(item), fmt),
              touchratio=data['x'][str(item)][1],
              ratio_low=data['touch_ratio_low'][str(item)],
              ratio_high=data['touch_ratio_hight'][str(item)],
              date_it=get_important_date(item),
              start_date_it=date(2020, 2, 15),
                )
        plot3_shade(
            data['sim_confirmed_deduction_s1'][str(item)], min_data['sim_confirmed_deduction_s1'][str(item)], max_data['sim_confirmed_deduction_s1'][str(item)],
            data['sim_confirmed_deduction_s2'][str(item)], min_data['sim_confirmed_deduction_s2'][str(item)], max_data['sim_confirmed_deduction_s2'][str(item)],
            data['sim_confirmed_deduction_s3'][str(item)], min_data['sim_confirmed_deduction_s3'][str(item)], max_data['sim_confirmed_deduction_s3'][str(item)],
            data['sim_confirmed_deduction_s8'][str(item)], min_data['sim_confirmed_deduction_s8'][str(item)],
            max_data['sim_confirmed_deduction_s8'][str(item)],
            data['real_confirmed'][str(item)],
              'Prediction of the cumulative number of confirmed cases',
              'img/{}/confirmed_prediction_shade.{}'.format(str(item), fmt),
              touchratio=data['x'][str(item)][1],
              ratio_low=data['touch_ratio_low'][str(item)],
              ratio_high=data['touch_ratio_hight'][str(item)],
              date_it=get_important_date(item),
              it_max=it_max,
              it_min=it_min,
              start_date_it=date(2020, 2, 15),
            loc='upper right')
        if item==900003:
            plot3_shade(
                data['sim_confirmed_deduction_s1'][str(item)], min_data['sim_confirmed_deduction_s1'][str(item)],
                max_data['sim_confirmed_deduction_s1'][str(item)],
                data['sim_confirmed_deduction_s2'][str(item)], min_data['sim_confirmed_deduction_s2'][str(item)],
                max_data['sim_confirmed_deduction_s2'][str(item)],
                data['sim_confirmed_deduction_s3'][str(item)], min_data['sim_confirmed_deduction_s3'][str(item)],
                max_data['sim_confirmed_deduction_s3'][str(item)],
                data['sim_confirmed_deduction_s8'][str(item)], min_data['sim_confirmed_deduction_s8'][str(item)],
                max_data['sim_confirmed_deduction_s8'][str(item)],
                data['real_confirmed'][str(item)],
                'Prediction of the cumulative number of confirmed cases',
                'img/{}/confirmed_prediction_shade.{}'.format(str(item), fmt),
                touchratio=data['x'][str(item)][1],
                ratio_low=data['touch_ratio_low'][str(item)],
                ratio_high=data['touch_ratio_hight'][str(item)],
                date_it=get_important_date(item),
                it_max=it_max,
                it_min=it_min,
                start_date_it=date(2020, 2, 1),
                loc='upper right')
        print('==================================',code_dict[item], str(final_date))
        print('final total infections and confirmed cases')
        print('confirmed')
        print('middle: ')
        print('100\%', format(int((data['sim_cum_confirmed_deduction_s1'][str(item)][-1])), ','),
              '({}-{})\n'.format(format(int((min_data['sim_cum_confirmed_deduction_s1'][str(item)][-1])),','),
                                 format(int((max_data['sim_cum_confirmed_deduction_s1'][str(item)][-1])),',')),
              '60\%', format(int((data['sim_cum_confirmed_deduction_s8'][str(item)])[-1]), ','),
              '({}-{})\n'.format(format(int((min_data['sim_cum_confirmed_deduction_s8'][str(item)][-1])),','),
                                 format(int((max_data['sim_cum_confirmed_deduction_s8'][str(item)][-1])),',')),
              '35\%', format(int((data['sim_cum_confirmed_deduction_s2'][str(item)])[-1]), ','),
              '({}-{})\n'.format(format(int((min_data['sim_cum_confirmed_deduction_s2'][str(item)][-1])),','),
                                 format(int((max_data['sim_cum_confirmed_deduction_s2'][str(item)][-1])),',')),
              '10\%', format(int((data['sim_cum_confirmed_deduction_s3'][str(item)])[-1]), ','),
              '({}-{})\n'.format(format(int((min_data['sim_cum_confirmed_deduction_s3'][str(item)][-1])),','),
                                 format(int((max_data['sim_cum_confirmed_deduction_s3'][str(item)][-1])),',')),)
        print('min: ')
        print('100\%', format(int((min_data['sim_cum_confirmed_deduction_s1'][str(item)][-1])), ','),
              '60\%', format(int((min_data['sim_cum_confirmed_deduction_s8'][str(item)])[-1]), ','),
              '35\%', format(int((min_data['sim_cum_confirmed_deduction_s2'][str(item)])[-1]), ','),
              '10\%', format(int((min_data['sim_cum_confirmed_deduction_s3'][str(item)])[-1]), ','),)
        print('max:')
        print('100\%', format(int((max_data['sim_cum_confirmed_deduction_s1'][str(item)][-1])), ','),
              '60\%', format(int((max_data['sim_cum_confirmed_deduction_s8'][str(item)])[-1]), ','),
              '35\%', format(int((max_data['sim_cum_confirmed_deduction_s2'][str(item)])[-1]), ','),
              '10\%', format(int((max_data['sim_cum_confirmed_deduction_s3'][str(item)])[-1]), ','),)
        print('infected')
        print('middle: ')
        print('100\%', format(int((data['sim_cum_infection_deduction_s1'][str(item)][-1])), ','),
              '({}-{})\n'.format(format(int((min_data['sim_cum_infection_deduction_s1'][str(item)][-1])), ','),
                                 format(int((max_data['sim_cum_infection_deduction_s1'][str(item)][-1])), ',')),
              '60\%', format(int((data['sim_cum_infection_deduction_s8'][str(item)])[-1]), ','),
              '({}-{})\n'.format(format(int((min_data['sim_cum_infection_deduction_s8'][str(item)][-1])), ','),
                                 format(int((max_data['sim_cum_infection_deduction_s8'][str(item)][-1])), ',')),
              '35\%', format(int((data['sim_cum_infection_deduction_s2'][str(item)])[-1]), ','),
              '({}-{})\n'.format(format(int((min_data['sim_cum_infection_deduction_s2'][str(item)][-1])), ','),
                                 format(int((max_data['sim_cum_infection_deduction_s2'][str(item)][-1])), ',')),
              '10\%', format(int((data['sim_cum_infection_deduction_s3'][str(item)])[-1]), ','),
              '({}-{})\n'.format(format(int((min_data['sim_cum_infection_deduction_s3'][str(item)][-1])), ','),
                                 format(int((max_data['sim_cum_infection_deduction_s3'][str(item)][-1])), ',')), )
        print('min: ')
        print('100\%', format(int((min_data['sim_cum_infection_deduction_s1'][str(item)][-1])), ','),
              '60\%', format(int((min_data['sim_cum_infection_deduction_s8'][str(item)])[-1]), ','),
              '35\%', format(int((min_data['sim_cum_infection_deduction_s2'][str(item)])[-1]), ','),
              '10\%', format(int((min_data['sim_cum_infection_deduction_s3'][str(item)])[-1]), ','),)
        print('max:')
        print('100\%', format(int((max_data['sim_cum_infection_deduction_s1'][str(item)][-1])), ','),
              '60\%', format(int((max_data['sim_cum_infection_deduction_s8'][str(item)])[-1]), ','),
              '35\%', format(int((max_data['sim_cum_infection_deduction_s2'][str(item)])[-1]), ','),
              '10\%', format(int((max_data['sim_cum_infection_deduction_s3'][str(item)])[-1]), ','),)
        print('==================================',code_dict[item], str(final_date))
    """
    plot3(get_total(data, 'sim_cum_confirmed_deduction_s1',total_cities),
          get_total(data, 'sim_cum_confirmed_deduction_s2',total_cities),
          get_total(data, 'sim_cum_confirmed_deduction_s3',total_cities),
          get_total(data, 'real_cum_confirmed',total_cities),
          'Prediction of the cumulative number of confirmed cases',
          'img/{}/cum_confirmed_prediction.{}'.format('0000', fmt),
          touchratio=1,
          ratio_low=0.5,
          ratio_high=1.5)
    plot3(get_total(data, 'sim_confirmed_deduction_s1', total_cities),
          get_total(data, 'sim_confirmed_deduction_s2', total_cities),
          get_total(data, 'sim_confirmed_deduction_s3', total_cities),
          get_total(data, 'real_confirmed', total_cities),
          'Prediction of the cumulative number of confirmed cases',
          'img/{}/confirmed_prediction.{}'.format('0000', fmt),
          touchratio=1,
          ratio_low=0.5,
          ratio_high=1.5)
    """

    print('drawing different deduction lately')
    for item in important_cities:
        print(item)
        it_max = None
        #if item == 900004:
        #    it_max = 40000
        #if item == 900005:
        #    it_max = 200000
        #if item == 900007:
        #    it_max = 80000
        #if item == 900008:
        #    it_max = 60000
        plot3(data['sim_cum_confirmed_deduction_s1'][str(item)],
              data['sim_cum_confirmed_deduction_s5'][str(item)],
              data['sim_cum_confirmed_deduction_s4'][str(item)],
              data['real_cum_confirmed'][str(item)],
              'Prediction of the cumulative number of confirmed cases',
              'img/{}/cum_confirmed_prediction_simdate.{}'.format(str(item), fmt),
              touchratio=data['x'][str(item)][1],
              ratio_low=data['touch_ratio_low'][str(item)],
              ratio_high=data['touch_ratio_hight'][str(item)],
              date_it=get_important_date(item) + timedelta(15),
              start_date_it=date(2020,2,15))
        plot3(data['sim_confirmed_deduction_s1'][str(item)],
              data['sim_confirmed_deduction_s5'][str(item)],
              data['sim_confirmed_deduction_s4'][str(item)],
              data['real_confirmed'][str(item)],
              'Prediction of the cumulative number of confirmed cases',
              'img/{}/confirmed_prediction_simdate.{}'.format(str(item), fmt),
              touchratio=data['x'][str(item)][1],
              ratio_low=data['touch_ratio_low'][str(item)],
              ratio_high=data['touch_ratio_hight'][str(item)],
              date_it=get_important_date(item) +timedelta(15) ,
              it_max=it_max,
              it_min=0,
              start_date_it=date(2020,2,15))
    """
    plot3(get_total(data, 'sim_cum_confirmed_deduction_s1', total_cities),
          get_total(data, 'sim_cum_confirmed_deduction_s5', total_cities),
          get_total(data, 'sim_cum_confirmed_deduction_s4', total_cities),
          get_total(data, 'real_cum_confirmed', total_cities),
          'Prediction of the cumulative number of confirmed cases',
          'img/{}/cum_confirmed_prediction_simdate.{}'.format('0000', fmt),
          touchratio=1,
          ratio_low=0.5,
          ratio_high=1.5)
    plot3(get_total(data, 'sim_confirmed_deduction_s1', total_cities),
          get_total(data, 'sim_confirmed_deduction_s5', total_cities),
          get_total(data, 'sim_confirmed_deduction_s4', total_cities),
          get_total(data, 'real_confirmed', total_cities),
          'Prediction of the cumulative number of confirmed cases',
          'img/{}/confirmed_prediction_simdate.{}'.format('0000', fmt),
          touchratio=1,
          ratio_low=0.5,
          ratio_high=1.5)
    """
    print('output loss')
    loss_dict = {}
    for item in total_cities:
        loss_dict[item] = data['loss'][str(item)]
    loss_df = pd.DataFrame(loss_dict, index=[0])
    loss_df.to_csv('./loss.csv', index=False)

    print('======== cur')
    for item in total_cities:
        print('{} &'.format(code_dict[item]), sep='', end='')
        print('{} &'.format(format(int(data['real_cum_confirmed'][str(item)][-1]), ',')), sep='',end='')
        print('{} ({}-{}) &'.format(
            format(int(data['sim_cum_confirmed_deduction_s1'][str(item)][lreal-1]), ','),
            format(int(min_data['sim_cum_confirmed_deduction_s1'][str(item)][lreal - 1 ]), ','),
            format(int(max_data['sim_cum_confirmed_deduction_s1'][str(item)][lreal - 1 ]), ','),
        ), sep='',end='')
        print('{} ({}-{}) &'.format(
            format(int(data['sim_cum_infection_deduction_s1'][str(item)][lreal-1]), ','),
            format(int(min_data['sim_cum_infection_deduction_s1'][str(item)][lreal - 1 ]), ','),
            format(int(max_data['sim_cum_infection_deduction_s1'][str(item)][lreal - 1]), ','),
        ), sep='',end='')
        print('{} ({}-{}) &'.format(
            format(int(data['sim_total_infection_deduction_s1'][str(item)][lreal-1]), ','),
            format(int(min_data['sim_total_infection_deduction_s1'][str(item)][lreal - 1]), ','),
            format(int(max_data['sim_total_infection_deduction_s1'][str(item)][lreal - 1 ]), ','),
        ), sep='',end='')
        print('{} ({}-{}) \\\\'.format(
            format(int(data['sim_cum_self_cured_deduction_s1'][str(item)][lreal-1]), ','),
            format(int(min_data['sim_cum_self_cured_deduction_s1'][str(item)][lreal - 1 ]), ','),
            format(int(max_data['sim_cum_self_cured_deduction_s1'][str(item)][lreal - 1 ]), ','),
        ), sep='',end='')
        print('')
    print('=======')
    # print variables
    x_list = construct_x(data, total_cities)
    min_x_list = construct_x(min_data, total_cities)
    max_x_list = construct_x(max_data, total_cities)
    format_out(x_list, min_x_list, max_x_list)


if __name__ == '__main__':
    main()
