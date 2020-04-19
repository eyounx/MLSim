import pandas as pd
import json
from datetime import datetime, timedelta, date
import os
import multiprocessing
import psutil
import os
import numpy as np
import random


def str2date(date):
    return datetime.strptime(date, "%Y-%m-%d").date()


# In fake data, Hubei data is China's Mainland data
def get_data_path(foreign_flag=False, fake_flag=False):
    if not foreign_flag:
        if fake_flag:
            return os.path.join(os.path.dirname(__file__), '../data/province_data_extra_3-13_fake.csv')
        return os.path.join(os.path.dirname(__file__), '../data/province_data_extra_3-13.csv')
    return os.path.join(os.path.dirname(__file__), '../data/data_4-15-foreign.csv')


def get_date_city(df, date, city, key='cum_confirmed'):
    value = df[(df['date']==date) & (df['adcode']==city)]
    if len(value) == 0:
        return 0
    else:
        return value[key].values[0]


# Hubei adcode:  420000    Wuhan adcode:  420100
def getCityDict(city_data: dict, transfer: pd.DataFrame):
    used_city = list(set(transfer['src_id']).union(set(transfer['dst_id'])))

    city_dict = {}
    for city in used_city:
        city_dict[city] = {
            'population': city_data[str(city)]['population'],
            'GDP': city_data[str(city)]['GDP']
        }

    return city_dict


def getCityFlow(transfer_data: pd.DataFrame, city_id, date):
    transfer_data = transfer_data[transfer_data['date'] == date]

    out_population = transfer_data[transfer_data['src_id'] == city_id]

    out_dict = {}
    for _, row in out_population.iterrows():
        out_dict[row['dst_id']] = row['travel_num']

    in_data = transfer_data[transfer_data['dst_id'] == city_id]

    in_dict = {}
    for _, row in in_data.iterrows():
        in_dict[row['src_id']] = row['travel_num']

    flow = {
        'out': sum(out_dict.values()),
        'in': in_dict
    }

    return flow


def flowHubei():
    if not os.path.exists(os.path.join(os.path.dirname(__file__), '../data/refine_transfer3.csv')):
        print('population movement file not found, all travel flow is set to 0!!!')
        return None
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/refine_transfer3.csv'))
    data = data[data['src_id'] == 420000]
    res = {}
    for _date, dst, num in zip(data['date'], data['dst_id'], data['travel_num']):
        date = datetime.strptime(_date, '%Y-%m-%d').date()
        if date not in res:
            res[date] = {}
        res[date][dst] = num
    return res




def flowOutData():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/refine_transfer3.csv'))
    cities = data['src_id'].unique()
    res = {}
    for city in cities:
        data_it = data[data['src_id'] == city]
        for _date, travel_num in zip(data_it['date'], data_it['travel_num']):
            date = datetime.strptime(_date, '%Y-%m-%d').date()
            if date not in res:
                res[date] = {}
            if city not in res[date]:
                res[date][city] = 0
            res[date][city] += travel_num
    return res


def flowInData():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/refine_transfer3.csv'))
    cities = data['dst_id'].unique()
    res = {}
    for city in cities:
        data_it = data[data['dst_id'] == city]
        for _date, travel_num in zip(data_it['date'], data_it['travel_num']):
            date = datetime.strptime(_date, '%Y-%m-%d').date()
            if date not in res:
                res[date] = {}
            if city not in res[date]:
                res[date][city] = 0
            res[date][city] += travel_num


def prepareData(old_data: pd.DataFrame):

    result = old_data.sort_values(by=['adcode', 'date'])
    return result

import numpy as np
def modifyTransferCSV():
    original_data = pd.read_csv('../data/refine_transfer.csv')
    cities = original_data['src_id'].unique()
    dates = original_data['date'].unique()
    min_dates = min(dates)
    max_dates = max(dates)
    data_map = {datea:
        {codea: {codeb: np.nan for codeb in cities}
                 for codea in cities} for datea in dates}

    for d in original_data['date'].unique():
        date_data = original_data[original_data['date'] == d]
        for codea, codeb, num in zip(date_data['src_id'], date_data['dst_id'], date_data['travel_num']):
            data_map[d][codea][codeb] = num
    for d in data_map:
        for codea in data_map[d]:
            for codeb in data_map[d][codea]:
                if np.isnan(data_map[d][codea][codeb]):
                    if codea == codeb:
                        data_map[d][codea][codeb] = 0
                        continue
                    date_now = datetime.strptime(d, '%Y-%m-%d').date()
                    dirct = True
                    while True:
                        # print(date_now)
                        if str(date_now) <= min_dates:
                            dirct = False
                        if dirct:
                            date_now = date_now - timedelta(1)
                        else:
                            date_now = date_now + timedelta(1)
                        if not np.isnan(data_map[str(date_now)][codea][codeb]):
                            data_map[d][codea][codeb] = data_map[str(date_now)][codea][codeb]
                            break
    res = {'date':[], 'src_id':[], 'dst_id': [], 'travel_num': []}
    for d in data_map:
        for codea in data_map[d]:
            for codeb in data_map[d][codea]:
                res['date'].append(d)
                res['src_id'].append(codea)
                res['dst_id'].append(codeb)
                res['travel_num'].append(data_map[d][codea][codeb])
    res_df = pd.DataFrame(res)
    res_df.sort_values(by=['date', 'src_id', 'dst_id'])
    res_df.to_csv('../data/refine_transfer3.csv')

def paramsToJson():
    data = pd.read_csv('../params.csv')
    x = [0.5270853640862626, 0.013463353896994268, 0.973943405909734, 0.09689887948044733, 0.009874877805477224, 8.000857830514523, 0.00010498914230194778, 0.17326532003033868]

    #[infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra]
    cities = list(data.keys().sort_values()[:-1])
    print(cities)
    res = {}
    for city in cities:
        res[city] = {'iso_rate': data[city][6],
                     'city_ctl': 1 - data[city][1]}
    #res['420000']['iso_rate'] = x[5]
    #res['420000']['city_ctl'] = 1 - x[1]
    print(res)
    with open('../initial_params.json', 'w') as f:
        f.write(json.dumps(res))

def modify_new_date(foreign_flag=False):
    data = pd.read_csv(get_data_path(foreign_flag))
    #data = pd.read_csv('../data/province_data_extra_2-17_increased.csv')
    d_list = []
    for i in range(len(data['date'])):
        d1 = datetime.strptime(data['date'][i], '%Y/%m/%d').date() + timedelta(days=1)
        d_list.append(str(d1))
        # data['date'][i] = str(d1)
    data.pop('date')
    data.insert(0,'date',d_list)
    #data.to_csv('../data/province_data_extra_2-17_increased.csv', index=False)
    data.to_csv(get_data_path(foreign_flag), index=False)

def add_newly_confirmed(foreign_flag=False):
    data = pd.read_csv(get_data_path(foreign_flag))
    d_list = []
    date_last = '2020-05-01'
    newly = []
    newly_dead = []
    newly_cured = []
    last = -1
    data = data.sort_values(by=['adcode','date'])
    data.to_csv(get_data_path(foreign_flag),index=False)
    data = pd.read_csv(get_data_path(foreign_flag))

    for i in range(len(data['date'])):
        d1 = datetime.strptime(data['date'][i], '%Y-%m-%d').date()
        d_list.append(str(d1))
        if str(d1) <= date_last:
            newly.append(0)
            newly_dead.append(0)
            newly_cured.append(0)
        else:
            error = data['cum_confirmed'][i] - last
            error = error if error >= 0 else 0
            newly.append(error)
            error = data['cum_dead'][i] - data['cum_dead'][i-1]
            error = error if error >= 0 else 0
            newly_dead.append(error)
            error = data['cum_cured'][i] - data['cum_cured'][i-1]
            error = error if error >= 0 else 0
            newly_cured.append(error)

        last = data['cum_confirmed'][i]
        date_last = str(d1)
        # data['date'][i] = str(d1)
    data.insert(0, 'observed', newly)
    data.insert(0, 'cured', newly_cured)
    data.insert(0, 'dead', newly_dead)
    data.to_csv(get_data_path(foreign_flag), index=False)
def test_yield(num):
    x,y,z = 0,0,0
    for i in range(num):
        x, y, z = yield (i, sum([x,y,z]))
        print(i, sum([x,y,z]))

def get_city_intensity(city):
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/city_intensity_2019_2020.csv'))
    intensity = list(data['intensity'].values)
    adcodes = list(data['adcode'].values)
    adcodes_new = [int(str(item)[:2]+'0000') for item in adcodes]
    data.insert(0, 'adcode_new', adcodes_new)
    data = data[(data['adcode_new'] == city) & (data['date'] >= '2020-01-01')]
    dates = list(data['date'].unique())
    res = {d: 0 for d in dates}
    for d in dates:
        res[d] += np.mean(list(data[data['date'] == d]['intensity']))
    return res

def merge_param():
    pro = [110000, 120000, 150000, 450000,
           500000, 540000, 620000, 630000, 640000, 650000]
    data_source = pd.read_csv('../params_source.csv')
    data_buding = pd.read_csv('../params_buding.csv')
    data_source_map = {}
    data_buding_map = {}
    keys =sorted(list(data_source.keys()))[:-1]
    for key in keys:
        data_source_map[int(key)] = list(data_source[key].values)

        #data_buding_map[int(key)] = list(data_buding[key].values)
    for key in pro:
        data_source_map[key] = list(data_buding[str(key)].values)
    res = pd.DataFrame(data_source_map)
    res.to_csv('../params.csv')

def merge_data():
    d1 = pd.read_csv('../data/province_data_extra_2-17_increased.csv')
    d2 = pd.read_csv('../data/province_data_extra_3-13.csv')
    d1 = d1.sort_values(by=['date', 'adcode'])
    d1.to_csv('./tmp.csv', index=False)
    d1 = pd.read_csv('./tmp.csv')
    d2 = d2.sort_values(by=['date', 'adcode'])
    d2.to_csv('./tmp.csv', index=False)
    d2 = pd.read_csv('./tmp.csv')
    for i in range(len(d1['date'])):
        if d1['adcode'][i] == 420000:
            print(d2['cum_confirmed'][i], d1['cum_confirmed'][i])
            d2['cum_confirmed'][i] = d1['cum_confirmed'][i]
            #d2['cum_cured'][i] = d1['cum_cured'][i]
            #d2['cum_dead'][i] = d1['cum_dead'][i]
            print(d2['adcode'][i], d1['adcode'][i])
            print(d2['date'][i], d1['date'][i])
            print('')
    d2.to_csv('../data/province_data_extra_3-13.csv', index=False)
    #for i in

def append_to_json(path, type, idx, data_in):
    data = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
    if type not in data:
        data[type] = {}
    data[type][idx] = data_in
    with open(path, 'w') as f:
        f.write(json.dumps(data))

def get_important_date(adcode):
    if adcode <= 900000:
        return date(2020, 1, 23)
    m = {
        #900004: date(2020, 3, 26), # USA
        900005: date(2020, 3, 10), # Italy
        #900006: date(2020, 3, 23), # UK
        #900007: date(2020, 3, 17), # France
        #900008: date(2020, 3, 17), # Germany
        900003: date(2020, 2, 25), # South Korea
        #900009: date(2020, 3, 17), # Iran
        #900010: date(2020, 3, 13), # Spain

        900004: date(2020, 3, 27), # USA
        #900005: date(2020, 3, 27), # Italy
        900006: date(2020, 3, 27), # UK
        900007: date(2020, 3, 27), # France
        900008: date(2020, 3, 27), # Germany
        #900003: date(2020, 3, 27), # South Korea
        900009: date(2020, 3, 27), # Iran
        900010: date(2020, 3, 27), # Spain

        900001: date(2020, 10, 1),
        900002: date(2020, 10, 1),
    }
    return m[adcode]

def it_code_dict():
    it_dict = {110000: 'Beijing', 120000: 'Tianjin', 130000: 'Hebei',
            140000: 'Shanxi', 150000: 'InnerMongoria', 210000: 'Liaoning',
            220000: 'Jilin', 230000: 'Heilongjiang', 310000: 'Shanghai',
            320000: 'Jiangsu', 330000: 'Zhejiang', 340000: 'Anhui',
            350000: 'Fujian', 360000: 'Jiangxi', 370000: 'Shandong',
            410000: 'Henan', 420000: 'Hubei', 430000: 'Hunan',
            440000: 'Guangdong', 450000: 'Guangxi', 460000: 'Hainan',
            500000: 'Chongqing', 510000: 'Sichuan', 520000: 'Guizhou',
            530000: 'Yunnan', 540000: 'Tibet', 610000: 'Shanxi',
            620000: 'Gansu', 630000: 'Qinghai', 640000: 'Ningxia',
            650000: 'Xinjiang', 710000: 'Taiwan', 810000: 'HongKong',
            820000: 'Macao', 900001: 'Singapore', 900002: 'Japan',
            900003: 'SouthKorea', 900004: 'USA', 900005: 'Italy',
            900006: 'UK', 900007: 'France', 900008: 'Germany', 900009: 'Iran',
            900010: 'Spain'}
    return it_dict


def codeDict():
    return it_code_dict()


def get_seed_num():
    return 10

def get_core_num():
    _res = multiprocessing.cpu_count() - 2
    if _res >= 38:
        _res = 38
    return _res

def clear_child_process():
    cur_process = psutil.Process()
    childs = cur_process.children()
    for item in childs:
        item.terminate()


def global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def get_death_rate(mortality, recovery, ob_period=25, cure_period=10):
    death = 0
    cur = 1
    for i in range(ob_period):
        new_death = cur * mortality
        death += cur * mortality
        cur -= new_death
        if i >= cure_period-1:
            new_recovery = cur * recovery
            cur -= new_recovery
    return death
# tmp = [(1 - x[6]) ** i * x[2] * x[1] for i in range(1, 15)]
#         x.append(np.sum(tmp))
#         DT = 1 / np.log2((1 + x[2] * x[1]) * (1 - x[6]))

def get_R0(touch_num, infect_ratio, quarantine, unob_period=14):
    return np.sum([(1 - quarantine) ** i * touch_num * infect_ratio for i in range(1, unob_period+1)])

def get_DT(touch_num, infect_ratio, quarantine):
    return 1 / np.log2((1 + touch_num * infect_ratio) * (1 - quarantine))

def DXY_data(date_it='2020-04-04 00:00:00'):
    data = pd.read_csv('DXYArea.csv')
    # got china
    data_china = data[data['countryEnglishName'] == 'China']
    code_dict = {110000: 'Beijing', 120000: 'Tianjin', 130000: 'Hebei',
            140000: 'Shanxi', 150000: 'InnerMongoria', 210000: 'Liaoning',
            220000: 'Jilin', 230000: 'Heilongjiang', 310000: 'Shanghai',
            320000: 'Jiangsu', 330000: 'Zhejiang', 340000: 'Anhui',
            350000: 'Fujian', 360000: 'Jiangxi', 370000: 'Shandong',
            410000: 'Henan', 420000: 'Hubei', 430000: 'Hunan',
            440000: 'Guangdong', 450000: 'Guangxi', 460000: 'Hainan',
            500000: 'Chongqing', 510000: 'Sichuan', 520000: 'Guizhou',
            530000: 'Yunnan', 540000: 'Tibet', 610000: 'Shanxi',
            620000: 'Gansu', 630000: 'Qinghai', 640000: 'Ningxia',
            650000: 'Xinjiang'}
    print(data_china)

    #prinvces = list(data_china['provinceEnglishName'].unique())
    prinvce_code = list(data_china['province_zipCode'].unique())
    prinvces = ['Neimenggu', 'Zhejiang', 'Guangdong', 'Hubei', 'Shanghai', 'Sichuan', 'Beijing', 'Fujian', 'Shandong', 'Liaoning', 'Hebei', 'Heilongjiang', 'Tianjin', 'Yunnan', 'Gansu', 'Shaanxi', 'Henan', 'Jiangsu', 'Shanxi', 'Hunan', 'Guangxi', 'Jiangxi', 'Jilin', 'Chongqing', 'Hainan', 'Guizhou', 'Ningxia', 'Xinjiang', 'Anhui', 'Qinghai', 'Xizang']
    province_data = {}
    province_confirmed = {}
    for p in prinvces:
        data_it = data_china[data_china['provinceEnglishName']==p]
        data_p = data_it['province_zipCode'].unique()
        data_tmp = data_it[['updateTime', 'province_zipCode', 'province_confirmedCount', 'province_suspectedCount', 'province_curedCount', 'province_deadCount']]
        data_time_tmp =  sorted(list(data_tmp['updateTime'].unique()))
        data_time_tmp = [item for item in data_time_tmp if item <= date_it][-1]
        province_data[data_p[0]] = data_tmp[data_tmp['updateTime'] == data_time_tmp][['updateTime', 'province_zipCode', 'province_confirmedCount', 'province_suspectedCount', 'province_curedCount', 'province_deadCount']].head(1)
        province_confirmed[int(data_p[0])] = list(province_data[data_p[0]]['province_confirmedCount'])[0]
        #print(province_data[data_p[0]])
        print(data_p[0], p, code_dict[data_p[0]])
        print(data_time_tmp)
        print(province_confirmed[data_p[0]])
    with open('./cur_confirmed-{}.json'.format(date_it[:10]), 'w') as f:
        f.write(json.dumps(province_confirmed))

def get_populations():
    res = {440000: 113460000.0, 370000: 100472400.0, 410000: 96050000.0,
           510000: 83410000.0,
     320000: 80507000.0, 130000: 75563000.0, 430000: 68988000.0, 340000: 63236000.0,
     420000: 59170000.0, 330000: 57370000.0, 450000: 49260000.0, 530000: 48005000.0,
     360000: 46476000.0, 210000: 43593000.0, 350000: 39410000.0, 610000: 38644000.0,
     230000: 37731000.0, 140000: 37183400.0, 520000: 36000000.0, 500000: 31017900.0,
     220000: 27040600.0, 620000: 26372600.0, 150000: 25340000.0, 650000: 24867600.0,
     310000: 24237800.0, 710000: 23690000.0, 110000: 21542000.0, 120000: 15596000.0,
     460000: 9343200.0, 810000: 7482500.0, 640000: 6881100.0, 630000: 6032300.0,
     540000: 3371500.0, 820000: 632000.0, 900001: 5612000, 900002: 126476461,
     900003: 51256670, 900004: 331002651, 900005: 60461826, 900006: 67890000,
     900007: 65273511, 900008: 83783942, 900009: 81160000, 900010: 46660000}
    return res

def construct_x(data, keys=None):
    x_list = []
    if keys is None:
        keys = data['x'].keys()
    code_dict = it_code_dict()
    for item in keys:
        x = data['x'][str(item)].copy()
        x.insert(0, code_dict[int(item)])
        x[2] *= 15
        x[7] *= 15
        x.append(data['newly_confirmed_loss'][str(item)])
        x.append(data['R01'][str(item)])
        x.append(data['DT1'][str(item)])
        x.append(data['R02'][str(item)])
        if data['DT2'][str(item)] > 0:
            x.append(data['DT2'][str(item)])
        else:
            x.append('-')
        x_list.append(x)
    return x_list


def format_out(x, min_x, max_x):
    labels = ['province', '$\\beta$', '$k$', '$\\gamma$', '$\\delta$', '$I(0)$', '$\\sigma$', '$k\'$', '$\\theta$', 'RMSE', '$R_0$', 'DT', '$R_0\'$', 'DT$\'$']
    for i in range(3):
        ranges = [[1,2,3,4], [5,6,7,8], [9, 10, 11, 12]]
        it_range = ranges[i]
        #it_range = [ii + 4 * i + 1 for ii in range(4)]
        #if i == 2:
        #    it_range.append(4 * 3 + 1)
        with open('./latex_param{}.txt'.format(i), 'w') as f:
            print('{} &'.format(labels[0]), sep='', end='')
            for ind in it_range:
                if ind == 0:
                    continue
                if ind >= len(labels):
                    break
                print('{} '.format(labels[ind]), sep='', end='')
                if not (ind == 0 or ind == len(x[0]) - 1 or ind == it_range[-1]):
                    print('&', sep='', end='')
            print('\\\\ \\hline')

            for ind2, line in enumerate(x):
                print('{} &'.format(line[0]), sep='', end='')
                # print('& ', sep='', end='')
                for ind, item in enumerate(line):
                    if ind not in it_range:
                        continue
                    if ind not in [0, 5]:
                        if not item == '-':
                            if isinstance(min_x[ind2][ind], str) or isinstance(max_x[ind2][ind], str):
                                print('{:.3f} ({}-{})'.format(item, min_x[ind2][ind], max_x[ind2][ind]),
                                      sep='', end='')
                            else:
                                print('{:.3f} ({:.3f}-{:.3f})'.format(item, min_x[ind2][ind], max_x[ind2][ind]),
                                      sep='', end='')
                        else:
                            print('{} '.format(item), sep='', end='')

                    else:
                        if ind == 0:
                            # continue
                            pass
                            # print('{} '.format(item), sep='', end='')
                        else:
                            print(
                                '{:.0f} ({}-{}) '.format(item, int(min_x[ind2][ind]), int(max_x[ind2][ind])),
                                sep='', end='')

                    if not ind == len(line) - 1 and not ind == 0 and not ind == it_range[-1]:
                        print('& ', sep='', end='')
                print('\\\\')
        print('\n\n')

def make_fake_data():
    data = pd.read_csv(get_data_path()).copy()
    cities = list(data['adcode'].unique())
    keys = ['dead', 'cured', 'observed', 'cum_cured', 'cum_dead', 'cum_confirmed']
    res = {}
    for item in cities:
        data_it = data[data['adcode'] == item]
        for key in keys:
            list_it = list(data_it[key])
            if key not in res:
                res[key] = [0] * len(list_it)
            for i in range(len(list_it)):
                res[key][i] += list_it[i]
    hubei_ind = list(data[data['adcode'] == 420000].index)
    for ind, ind_data in enumerate(hubei_ind):
        for key in keys:
            data.loc[ind_data, (key)] = res[key][ind]
    data.to_csv(get_data_path()[:-4]+'_fake.csv', index=False)

def clip_time(date_it,foreign_flag=True):
    data = pd.read_csv(get_data_path(foreign_flag=foreign_flag))
    data = data[data['date']<=str(date_it)]
    data.to_csv(get_data_path(foreign_flag=foreign_flag), index=False)

if __name__ == '__main__':
    #make_fake_data()
    #exit(0)
    forein_flag = True
    modify_new_date(forein_flag)
    #merge_data()
    add_newly_confirmed(forein_flag)

