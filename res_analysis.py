from components.utils import get_seed_num, it_code_dict
import numpy as np
import json


def res_analysis(data_buffer):
    max_data = {}
    min_data = {}
    mean_data = {}
    std_data = {}
    best_data = {}
    middle_data = {}
    xmin_data = {}
    xmax_data = {}
    xmean_data = {}
    xstd_data = {}
    xind = 1
    if len(data_buffer) <= 3:
        xind = 0
    newly_confirmed_loss_dict = [data_buffer[i]['newly_confirmed_loss'] for i in range(len(data_buffer))]
    loss_dict = [data_buffer[i]['loss'] for i in range(len(data_buffer))]
    loss_ind = {}
    for sub_key in newly_confirmed_loss_dict[0].keys():
        cur_loss = [newly_confirmed_loss_dict[i][sub_key] for i in range(len(data_buffer))]
        loss_ind[sub_key] = int(np.argmin(cur_loss))
    # loss_ind = int(np.argmin(loss))
    for key in data_buffer[0].keys():
        if not key in max_data:
            max_data[key] = {}
            min_data[key] = {}
            mean_data[key] = {}
            std_data[key] = {}
            best_data[key] = {}
            middle_data[key] = {}
            xmin_data[key] = {}
            xmax_data[key] = {}
            xmean_data[key] = {}
            xstd_data[key] = {}

        for sub_key in data_buffer[0][key].keys():
            if isinstance(data_buffer[0][key][sub_key], list):
                max_list = []
                min_list = []
                mean_list = []
                std_list = []
                middle_list = []
                xmin_list = []
                xmax_list = []
                xmean_list = []
                xstd_list = []
                best_list = data_buffer[loss_ind[sub_key]][key][sub_key].copy()
                for ind in range(len(data_buffer[0][key][sub_key])):
                    it_data = [data_buffer[i][key][sub_key][ind] for i in range(len(data_buffer))]
                    max_list.append(float(np.max(it_data)))
                    min_list.append(float(np.min(it_data)))
                    mean_list.append(float(np.mean(it_data)))
                    std_list.append(float(np.std(it_data)))
                    middle_list.append(float(np.median(it_data)))
                    it_data_sorted = np.sort(it_data)
                    xmin_list.append(float(it_data_sorted[xind]))
                    xmax_list.append(float(it_data_sorted[-(xind+1)]))
                    xmean_list.append(float(np.mean(it_data_sorted[xind:-xind])))
                    xstd_list.append(float(np.std(it_data_sorted[xind:-xind])))

                max_data[key][sub_key] = max_list.copy()
                min_data[key][sub_key] = min_list.copy()
                mean_data[key][sub_key] = mean_list.copy()
                best_data[key][sub_key] = best_list.copy()
                std_data[key][sub_key] = std_list.copy()
                middle_data[key][sub_key] = middle_list.copy()
                xmin_data[key][sub_key] = xmin_list.copy()
                xmax_data[key][sub_key] = xmax_list.copy()
                xmean_data[key][sub_key] = xmean_list.copy()
                xstd_data[key][sub_key] = xstd_list.copy()

            elif isinstance(data_buffer[0][key][sub_key], int) or isinstance(data_buffer[0][key][sub_key], float):
                it_data = [data_buffer[i][key][sub_key] for i in range(len(data_buffer))]
                best_it = data_buffer[loss_ind[sub_key]][key][sub_key]
                max_it = float(np.max(it_data))
                min_it = float(np.min(it_data))
                mean_it = float(np.mean(it_data))
                std_it = float(np.std(it_data))
                middle_it = (float(np.median(it_data)))
                it_data_sorted = np.sort(it_data)
                xmin_it = (float(it_data_sorted[xind]))
                xmax_it = (float(it_data_sorted[-(xind+1)]))
                xmean_it = (float(np.mean(it_data_sorted[xind:-xind])))
                xstd_it = (float(np.std(it_data_sorted[xind:-xind])))

                max_data[key][sub_key] = max_it
                min_data[key][sub_key] = min_it
                mean_data[key][sub_key] = mean_it
                best_data[key][sub_key] = best_it
                std_data[key][sub_key] = std_it
                middle_data[key][sub_key] = middle_it
                xmin_data[key][sub_key] = xmin_it
                xmax_data[key][sub_key] = xmax_it
                xmean_data[key][sub_key] = xmean_it
                xstd_data[key][sub_key] = xstd_it
            else:
                assert 'type error'
    return max_data, min_data, mean_data, best_data, std_data, middle_data, xmin_data, xmax_data, xmean_data, xstd_data

def load_and_save(input_format, output_format, num=None, start_ind=0):
    if num is None:
        num = get_seed_num()
    data_buffer = [json.load(open(input_format.format(int(i)), 'r')) for i in range(start_ind, num + start_ind)]
    max_data, min_data, mean_data, best_data, std_data, middle_data, xmin_data, xmax_data, xmean_data, xstd_data = res_analysis(
        data_buffer)
    with open(output_format.format('min'), 'w') as f:
        f.write(json.dumps(min_data))

    with open(output_format.format('max'), 'w') as f:
        f.write(json.dumps(max_data))

    with open(output_format.format('mean'), 'w') as f:
        f.write(json.dumps(mean_data))

    with open(output_format.format('best'), 'w') as f:
        f.write(json.dumps(best_data))

    with open(output_format.format('std'), 'w') as f:
        f.write(json.dumps(std_data))

    with open(output_format.format('middle'), 'w') as f:
        f.write(json.dumps(middle_data))
    with open(output_format.format('xmin'), 'w') as f:
        f.write(json.dumps(xmin_data))
    with open(output_format.format('xmax'), 'w') as f:
        f.write(json.dumps(xmax_data))
    with open(output_format.format('xmean'), 'w') as f:
        f.write(json.dumps(xmean_data))
    with open(output_format.format('xstd'), 'w') as f:
        f.write(json.dumps(xstd_data))
    return  max_data, min_data, mean_data, best_data, std_data, middle_data, xmin_data, xmax_data, xmean_data, xstd_data

if __name__ == '__main__':
    data_buffer = [json.load(open('./data_run_foreign{}.json'.format(int(i)), 'r')) for i in range(get_seed_num())]
    cities = [900003, 900004, 900005, 900006, 900007, 900008, 900009, 900010]
    #data_buffer = [json.load(open('./data_run{}.json'.format(int(i)), 'r')) for i in range(get_seed_num())]
    #cities = [420000]
    max_data, min_data, mean_data, best_data, std_data, middle_data, xmin_data, xmax_data, xmean_data, xstd_data = res_analysis(data_buffer)
    pbnum = 3
    for iitem in cities:
        city = str(iitem)
        print(it_code_dict()[int(city)])
        x = [data_buffer[i]['x'][city] for i in range(get_seed_num())]
        final_cum = [data_buffer[i]['sim_cum_confirmed_deduction_s1'][city][-1] for i in range(get_seed_num())]
        final_cum_infection = [data_buffer[i]['sim_cum_infection_deduction_s1'][city][-1] for i in range(get_seed_num())]
        # self.models = [infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra, base_touch_num, cure_ratio]
        print('info, \t\t\t', 'infec\ttouch\tobs\tdead1\tdummy\tiso\t-\tcure,\t\ttotal confirmed\t\tratio', sep='')
        for ind, item in enumerate(x):
            print(ind, '\t', *list(np.round(item, pbnum)), final_cum[ind], '\t', np.round((1- final_cum[ind] / final_cum_infection[ind])*100, 2), sep='\t')
        print('max, \t\t', *list(np.round(max_data['x'][city], pbnum)), sep='\t')
        print('min, \t\t', *list(np.round(min_data['x'][city], pbnum)), sep='\t')
        print('mean, \t\t', *list(np.round(mean_data['x'][city], pbnum)), sep='\t')
        print('mid, \t\t', *list(np.round(middle_data['x'][city], pbnum)), sep='\t')
        print('std, \t\t', *list(np.round(std_data['x'][city], pbnum)), sep='\t')
        print('best, \t\t', *list(np.round(best_data['x'][city], pbnum)), sep='\t')
        print('best loss, \t', (np.round(best_data['newly_confirmed_loss'][city], pbnum)), sep='\t')
        print('')
