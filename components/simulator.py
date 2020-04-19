from datetime import timedelta, datetime
import math, logging, os, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), "../../ncovmodel"))
from zoopt import Dimension, Objective, Parameter, ExpOpt
import pandas as pd
import numpy as np
from components.utils import get_populations

log = logging.getLogger("Simluator")


class PseudoSolution():

    def __init__(self, x):
        self.x = x

    def get_x(self):
        return self.x


class InfectionList:
    def __init__(self, dummy, init_infection_ratio, ob, isoratio, incubation=3, base_touch_num=15,
                 unob_period=30, population=1000, init_unob_pre=None,
                 obs_period=25, iso_period=10, cure_period=10, no_healed=False):
        self.observed_period = obs_period
        self.isolation_period = iso_period
        self.unob_period = unob_period #14
        self.cured_period = cure_period
        self.infection_list_len = max(self.observed_period, self.isolation_period, self.unob_period)
        self.incubation = incubation
        self.ob = ob
        self.no_healed = no_healed
        #self.whole_period = whole_period
        self.isoratio = isoratio
        self.population = population
        self.cum_infection = 0
        self.infection_list = [0] * self.infection_list_len
        self.observed_list = [0] * self.observed_period
        self.intra_observed_list = [0] * self.observed_period
        self.isolation_list = [0] * self.isolation_period
        self.intra_isolation_list = [0] * self.isolation_period
        #dummy = dummy * ((1 + init_infection_ratio) ** 7)
        self.unobserved_list = [dummy]
        self.unob_intra_list = [0]
        self.total_infection = dummy
        self.total_ob_infection = 0
        self.total_unob_infection = dummy
        self.total_intra_unob = 0
        self.total_isolation = 0
        self.cumulative_observed = 0
        self.cumulative_intra_observed = 0
        self.cum_cured = 0
        self.cum_self_cured = 0
        self.ob_dead_list = [0]
        self.unob_dead_list = [0]
        self.total_ob_dead = 0
        self.total_dead = 0
        self.out_ratio = 1
        self.new_infection = 0
        for i in range(self.unob_period):
            self.unob_intra_list.append(0)
        if init_unob_pre:
            for i in range(self.unob_period):
                self.unob_intra_list[i + 1] += (init_unob_pre[i])
        for i in range(self.unob_period):
            infection_num = (np.sum(self.unobserved_list) + np.sum(self.unob_intra_list[:i+1])) * init_infection_ratio * base_touch_num
            self.unobserved_list.append(infection_num * (1 - np.sum(self.unobserved_list) / self.population))
        for ii in range(incubation+1):
            t_cum = 0
            for i in range(self.unob_period - incubation + ii):
                suspect = self.unobserved_list[i] * isoratio
                self.unobserved_list[i] -= suspect
                t_cum += suspect
            self.isolation_list.append(t_cum)
            self.isolation_list.pop(0)
            t_cum = 0
            for i in range(self.unob_period - incubation + ii):
                suspect = self.unob_intra_list[i] * isoratio
                self.unob_intra_list[i] -= suspect
                t_cum += suspect
            self.intra_isolation_list.append(t_cum)
            self.intra_isolation_list.pop(0)
        t_cum = 0
        t_intra_cum = 0
        for i in range(self.isolation_period - incubation):
            confirm = self.isolation_list[i] * ob
            intra_confirm = self.intra_isolation_list[i] * ob
            t_cum += confirm
            t_intra_cum += intra_confirm
            self.isolation_list[i] -= confirm
            self.intra_isolation_list[i] -= intra_confirm
        self.observed_list.append(t_cum)
        self.observed_list.pop(0)
        self.intra_observed_list.append(t_intra_cum)
        self.intra_observed_list.pop(0)
        self.unobserved_list.pop(0)
        self.unob_intra_list.pop(0)
        #self.cum_isolation

        for i in range(self.infection_list_len):
            self.infection_list[self.infection_list_len - i - 1] = 0
            if i < self.observed_period:
                self.infection_list[self.infection_list_len - i - 1] += self.observed_list[self.observed_period - i - 1]
                self.infection_list[self.infection_list_len - i - 1] += self.intra_observed_list[self.observed_period - i - 1]
            if i < self.isolation_period:
                self.infection_list[self.infection_list_len - i - 1] += self.isolation_list[self.isolation_period - i - 1]
                self.infection_list[self.infection_list_len - i - 1] += self.intra_isolation_list[self.isolation_period - i - 1]
            if i < self.unob_period:
                self.infection_list[self.infection_list_len - i - 1] += self.unobserved_list[self.unob_period - i - 1]
                self.infection_list[self.infection_list_len - i - 1] += self.unob_intra_list[self.unob_period - i - 1]

        #self.infection_list[-1] += self.observed_list[-1]
        self.total_infection = np.sum(self.infection_list)
        self.total_ob_infection = np.sum(self.observed_list)
        self.total_unob_infection =np.sum(self.unobserved_list) + np.sum(self.unob_intra_list)
        self.total_intra_unob = np.sum(self.unob_intra_list)
        self.total_isolation = np.sum(self.isolation_list)
        self.cumulative_observed += max(0, self.observed_list[-1] + self.intra_observed_list[-1])
        self.cumulative_intra_observed += max(0, self.intra_observed_list[-1])
        self.unob_init = [item1 + item2 for item1, item2 in zip(self.unobserved_list, self.unob_intra_list)]
        self.unob_init.pop(-1)
        self.unob_total_init = np.cumsum(self.unob_init).tolist()
        self.cured = 0
        self.sel_cured = 0
        self.cum_infection += np.sum(self.infection_list)

    def update(self, new_infection, intra_new_infect, dead_rate_ob, dead_rate_unob, cure_ratio, out_population=0):
        new_infection = min(self.population / 10, new_infection)
        intra_new_infect = min(self.population / 10, intra_new_infect)
        t_ob_dead = 0
        t_unob_dead = 0
        #average_out = out_infection / self.population * se / self.whole_period
        t_confirm = 0
        t_confirm_intra = 0
        t_suspect = 0
        t_suspect_intra = 0
        t_iso_dead = 0
        t_cure = 0
        t_cure_self = 0
        # unob
        interp_flag = False
        olen = self.isolation_period - self.incubation
        if interp_flag:
            confirm_ratio_interp = [((i + 1) / olen) * (1 - self.ob) + self.ob for i in reversed(range(olen))]
        else:
            confirm_ratio_interp = [self.ob] * olen
        olen = self.observed_period - self.cured_period
        if interp_flag:
            cure_ratio_interp = [((i + 1) / olen)*(1-cure_ratio) + cure_ratio for i in reversed(range(olen))]
        else:
            cure_ratio_interp = [cure_ratio] * olen
        for i in range(self.unob_period):
            suspect = self.unobserved_list[i] * self.isoratio #* np.sin(i / self.unob_period * np.pi)
            intra_suspect =self.unob_intra_list[i] * self.isoratio #* np.sin(i / self.unob_period * np.pi)
            flow_out_it = self.unobserved_list[i] * out_population / self.population * self.out_ratio
            self.unobserved_list[i] = max(self.unobserved_list[i] - suspect - flow_out_it, 0)
            flow_out_it = self.unob_intra_list[i] * out_population / self.population * self.out_ratio
            self.unob_intra_list[i] = max(self.unob_intra_list[i] - flow_out_it - intra_suspect, 0)
            t_suspect += suspect
            t_suspect_intra += intra_suspect
        for i in range(self.observed_period):
            ob_dead = self.observed_list[i] * dead_rate_ob
            self.observed_list[i] -= ob_dead
            ob_dead_intra = self.intra_observed_list[i] * dead_rate_ob
            self.intra_observed_list[i] -= ob_dead_intra
            t_ob_dead += (ob_dead + ob_dead_intra)

        for i in range(self.observed_period - self.cured_period):
            ob_cure = self.observed_list[i] * cure_ratio_interp[i]
            self.observed_list[i] -= ob_cure
            ob_cure_intra = self.intra_observed_list[i] * cure_ratio_interp[i]
            self.intra_observed_list[i] -= ob_cure_intra
            t_cure += (ob_cure_intra + ob_cure)
        t_cure += self.intra_observed_list[0]
        self.intra_observed_list[0] = 0
        t_cure += self.observed_list[0]
        self.observed_list[0] = 0

        for i in range(self.unob_period):
            unob_dead = self.unobserved_list[i] * dead_rate_unob
            unob_dead_intra = self.unob_intra_list[i] * dead_rate_unob
            self.unobserved_list[i] = max(self.unobserved_list[i] - unob_dead, 0)
            self.unob_intra_list[i] = max(self.unob_intra_list[i] - unob_dead_intra, 0)
            t_unob_dead += unob_dead + unob_dead_intra

        if self.no_healed:
            t_suspect += self.unobserved_list[0]
            t_suspect_intra += self.unob_intra_list[0]
            self.unobserved_list[0] = 0
            self.unob_intra_list[0] = 0

        for i in range(self.isolation_period - self.incubation):
            # update isolation
            confirm = self.isolation_list[i] * confirm_ratio_interp[i]
            confirm_intra = self.intra_isolation_list[i] * confirm_ratio_interp[i]
            iso_dead = self.isolation_list[i] * dead_rate_unob
            iso_dead_intra = self.intra_isolation_list[i] * dead_rate_unob
            self.isolation_list[i] = max(self.isolation_list[i] - confirm - iso_dead, 0)
            self.intra_isolation_list[i] = max(self.intra_isolation_list[i] - confirm_intra - iso_dead_intra, 0)
            # update infection_list
            t_confirm += confirm
            t_iso_dead += iso_dead + iso_dead_intra
            t_confirm_intra += confirm_intra
        t_confirm += self.isolation_list[0]
        t_confirm_intra += self.intra_isolation_list[0]
        self.isolation_list[0] = 0
        self.intra_isolation_list[0] = 0

        for i in range(self.infection_list_len):
            self.infection_list[self.infection_list_len - i - 1] = 0
            if i < self.observed_period:
                self.infection_list[self.infection_list_len - i - 1] += self.observed_list[self.observed_period - i - 1]
                self.infection_list[self.infection_list_len - i - 1] += self.intra_observed_list[
                    self.observed_period - i - 1]
            if i < self.isolation_period:
                self.infection_list[self.infection_list_len - i - 1] += self.isolation_list[
                    self.isolation_period - i - 1]
                self.infection_list[self.infection_list_len - i - 1] += self.intra_isolation_list[
                    self.isolation_period - i - 1]
            if i < self.unob_period:
                self.infection_list[self.infection_list_len - i - 1] += self.unobserved_list[self.unob_period - i - 1]
                self.infection_list[self.infection_list_len - i - 1] += self.unob_intra_list[self.unob_period - i - 1]

        self.observed_list.append(t_confirm)
        self.new_infection = new_infection + intra_new_infect
        self.intra_observed_list.append(t_confirm_intra)
        self.isolation_list.append(t_suspect)
        self.intra_isolation_list.append(t_suspect_intra)
        self.unobserved_list.append(new_infection)
        self.unob_intra_list.append(intra_new_infect)
        self.infection_list.append(new_infection + intra_new_infect + t_suspect_intra + t_suspect + t_confirm + t_confirm_intra)
        self.cured = t_cure
        self.cum_infection += (intra_new_infect + new_infection)
        t_cure_self += self.unobserved_list.pop(0)
        self.infection_list.pop(0)
        t_cure_self += self.unob_intra_list.pop(0)
        self.isolation_list.pop(0)
        self.observed_list.pop(0)
        self.intra_observed_list.pop(0)
        self.intra_isolation_list.pop(0)
        self.sel_cured = t_cure_self
        self.total_infection = np.sum(self.infection_list)
        self.total_ob_infection = np.sum(self.observed_list)
        self.total_unob_infection = np.sum(self.unobserved_list) + np.sum(self.unob_intra_list)
        self.total_intra_unob = np.sum(self.unob_intra_list)
        self.total_isolation = np.sum(self.isolation_list)
        self.cumulative_observed += max(self.observed_list[-1] + self.intra_observed_list[-1], 0)
        self.cumulative_intra_observed += max(self.intra_observed_list[-1], 0)

        self.ob_dead_list.append(t_ob_dead)
        self.unob_dead_list.append(t_unob_dead)
        self.total_ob_dead += t_ob_dead
        self.total_dead += (t_ob_dead + t_unob_dead + t_iso_dead)
        self.cum_cured += max(self.cured,0)
        self.cum_self_cured += max(self.sel_cured, 0)
        #print(self.cum_infection , self.cum_self_cured + self.total_infection)


def date2str(date):
    return date.strftime("%Y-%m-%d")


def gen_date_list(start_date, duration):
    date = datetime.strptime(start_date, "%Y-%m-%d").date()
    date_list = [date]
    for i in range(duration):
        date_list.append(date_list[-1] + timedelta(days=1))
    return date_list[0], date_list[1:]

def get_loss(sim_res, real_res, ord=0., start_date=None, end_date=None):
    sim_res = sim_res.sort_values(by=["adcode", "date", ])
    real_res = real_res.sort_values(by=["adcode", "date"])
    if start_date is not None:
        sim_res = sim_res[sim_res['date'] >= str(start_date)]
        real_res = real_res[real_res['date'] >= str(start_date)]
    if end_date is not None:
        sim_res = sim_res[sim_res['date'] <= str(end_date)]
        real_res = real_res[real_res['date'] <= str(end_date)]

    columns = ["observed", 'cured', 'dead']
    w = [1.] * len(columns)
    diff = 0
    if len(sim_res[columns[0]]) == 0:
        return 0
    for ind, column in enumerate(columns):
        if len(sim_res[column]) != len(real_res[column]):
            raise Exception(f"Sim result size: {len(sim_res[column])}, real res size: {len(real_res[column])}.")

        d1 = list(sim_res[column].values)
        d2 = list(real_res[column].values)
        r = [(item+1) / len(d1) for item in range(len(d1))]
        _diff = 0
        _den = 0
        for i in range(len(d1)):
            _ratio = (r[i] ** ord)
            _diff += ((d1[i] - d2[i]) ** 2) * _ratio
            _den += _ratio
        _diff /= _den
        diff += _diff * w[ind]
    return math.sqrt(diff)


def get_newly_loss(sim_res, real_res, start_date=None, end_date=None):
    sim_res = sim_res.sort_values(by=["adcode", "date", ])
    real_res = real_res.sort_values(by=["adcode", "date"])
    if start_date is not None:
        sim_res = sim_res[sim_res['date'] >= str(start_date)]
        real_res = real_res[real_res['date'] >= str(start_date)]
    if end_date is not None:
        sim_res = sim_res[sim_res['date'] <= str(end_date)]
        real_res = real_res[real_res['date'] <= str(end_date)]
    columns = ["observed"]
    w = [1.] * len(columns)
    diff = 0
    if len(sim_res[columns[0]]) == 0:
        return 0
    for ind, column in enumerate(columns):
        if len(sim_res[column]) != len(real_res[column]):
            raise Exception(f"Sim result size: {len(sim_res[column])}, real res size: {len(real_res[column])}.")

        d1 = list(sim_res[column].values)
        d2 = list(real_res[column].values)
        r = [(item+1) / len(d1) for item in range(len(d1))]
        _diff = 0
        _den = 0
        for i in range(len(d1)):
            _ratio = (r[i] ** 0.)
            _diff += ((d1[i] - d2[i]) ** 2) * _ratio
            _den += _ratio
        _diff /= _den
        diff += _diff * w[ind]
    return math.sqrt(diff)

def cal_infect_ratio(infection, population):
    assert set(infection.keys()) == set(population.keys()), 'ADCODE MISMATCH'
    infect_ratio = {}
    for city in infection.keys():
        if float(population[city]) < 1e-10:
            infect_ratio[city] = 0
        else:
            infect_ratio[city] = float(infection[city].total_infection) / float(population[city])
    return infect_ratio


def gen_date_result(city, date, infection, inc_incity, round_result=False):
    record = {'date': date2str(date)}
    record_date = {}

    def post_handle(value):
        if round_result:
            return round(value)
        else:
            return value

    record_date[city] = {
        # infection
        'observed_infection': post_handle(infection[city].observed_list[-1]),
        'unobserved_infection': post_handle(infection[city].unobserved_list[-1]),
        'infection': post_handle(infection[city].total_infection),
        'cum_observed_infection': post_handle(infection[city].cumulative_observed),
        # dead
        'ob_dead': post_handle(infection[city].ob_dead_list[-1]),
        'unob_dead': post_handle(infection[city].unob_dead_list[-1]),
        'cum_ob_dead': post_handle(infection[city].total_ob_dead),
        'cum_dead': post_handle(infection[city].total_ob_dead),
        # Origin of the increasing number of patients
        'increased_in_city': post_handle(inc_incity[city]),
    }
    record['record'] = record_date

    return record


def gen_history(city, date, infection, round_result=False):
    predict_records = []

    def post_handle(value):
        if round_result:
            return round(value)
        else:
            return value

    predict_record = {}
    predict_record["adcode"] = city
    predict_record["date"] = str(date)
    predict_record['cum_confirmed'] = post_handle(infection[city].cumulative_observed)
    predict_record['cum_confirmed_intra_city'] = post_handle(infection[city].cumulative_intra_observed)
    predict_record['observed_dead'] = post_handle(infection[city].ob_dead_list[-1])
    predict_record['observed'] = post_handle(infection[city].observed_list[-1] +infection[city].intra_observed_list[-1] )
    predict_record['observed_infection'] = post_handle(infection[city].observed_list[-1])
    predict_record['total_unobserved'] = post_handle(infection[city].total_unob_infection)
    predict_record['total_infection'] = post_handle(infection[city].total_infection)
    predict_record['cum_infection'] = post_handle(infection[city].cum_infection)
    predict_record['isolation'] = post_handle(infection[city].isolation_list[-1])
    predict_record['total_isolation'] = post_handle(infection[city].total_isolation)
    predict_record['cum_dead'] = post_handle(infection[city].total_dead)
    predict_record['new_infection'] = post_handle(infection[city].new_infection)

    predict_record['cured'] = post_handle(infection[city].cured)
    predict_record['dead'] = post_handle(infection[city].ob_dead_list[-1] + infection[city].unob_dead_list[-1])
    predict_record['self_cured'] = post_handle(infection[city].sel_cured)
    predict_record['cum_cured'] = post_handle(infection[city].cum_cured)
    predict_record['cum_self_cured'] = post_handle(infection[city].cum_self_cured)
    predict_record['out_in_population'] = post_handle(infection[city].unob_intra_list[-1])
    predict_record['observed_intra'] = post_handle(infection[city].intra_observed_list[-1])
    predict_record['cum_no_symbol'] = post_handle(infection[city].cum_self_cured + infection[city].total_unob_infection)

    predict_records.append(predict_record)
    return predict_records


def problem_maker(simulator, real_result, date_end=None, ord=0):
    def problem(solution):
        x = solution.get_x()
        simulator.set_param(x)
        start_date = datetime.strptime(real_result["date"].min(), "%Y-%m-%d").date()
        duration = len(real_result["date"].unique()) - 1
        # print('start_data ', start_date, 'duration: ', duration)
        sim_res, _ = simulator.simulate(str(start_date), duration)
        if date_end is None:
            return get_loss(sim_res, real_result, ord)
        else:
            return get_loss(sim_res[sim_res['date'] <= str(date_end)],
                            real_result[real_result['date'] <= str(date_end)], 2)
    return problem


class Simulator:
    def __init__(self, city, infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra, cure_ratio,
                 important_dates=None, incubation=3,
                 unob_flow_num=None, flow_out_data=None, training_date_end=None,
                 unob_period=14, obs_period=25, iso_period=10, cure_period=10, no_healed=False):
        self.city = city
        self.unob_period = unob_period
        self.obs_period = obs_period
        self.iso_period = iso_period
        self.cure_period = cure_period
        self.incubation = 0
        self.history = []
        self.no_healed = no_healed
        self.population_map = get_populations()
        if city in self.population_map:
            self.population = self.population_map[city]
        else:
            self.population = 1000000000
        self.flow_out_data = flow_out_data
        if important_dates is None:
            change_date = datetime.strptime('2020-1-23', '%Y-%m-%d').date()
            self.important_dates = [change_date]
        else:
            self.important_dates = important_dates.copy()
        assert len(self.important_dates) == touchratio.ndim
        # models
        self.infectratio = infectratio
        self.touchratio = touchratio
        self.base_touch_num = 15
        self.cure_ratio = cure_ratio
        self.obs = obs
        self.dead = dead
        self.dummy = dummy
        self.isoratio = isoratio
        self.touchratiointra = touchratiointra
        self.models = [infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra, cure_ratio]

        self.unob_flow_num = unob_flow_num
        dim = 0
        dim_range = []
        dim_type = []
        self.dims = [0]
        self.training_date_end = training_date_end
        for m in self.models:
            log.info(f'{m.__class__.__name__} {m.get_param_info()} {m.params}')
            d_dim, d_range, d_type = m.get_param_info()
            dim += d_dim
            self.dims.append(dim)
            dim_range += d_range
            dim_type += d_type
        self.dim = dim
        self.dim_range = dim_range
        self.lower = [d[0] for d in dim_range]
        self.upper = [d[1] for d in dim_range]
        self.dim_type = dim_type
        self.param = []
        log.info(f'dim = {self.dim}')
        log.info(f'dim_range = {self.dim_range}')
        log.info(f'dim_type = {self.dim_type}')
        log.info(f'dims = {self.dims}')

    def set_period(self, unob_period=None, obs_period=None, iso_period=None, cure_period=None):
        if unob_period is not None:
            self.unob_period = unob_period
        if obs_period is not None:
            self.obs_period = obs_period
        if iso_period is not None:
            self.iso_period = iso_period
        if cure_period is not None:
            self.cure_period = cure_period

    def fit(self, real_data, budget=10000, server_num=3, repeat=1, seed=1, plot=False, plot_file="optimize.png",
            intermediate_freq=100, init_samples=None, loss_ord=0):
        problem = problem_maker(self, real_data, self.training_date_end, loss_ord)
        dim, dim_range, dim_type = self.get_dim()  # dimension
        dim_range = [[a[0], a[1]] for a in dim_range]
        print(dim_range)
        objective = Objective(problem, Dimension(dim, dim_range, dim_type))  # set up objective
        parameter = Parameter(algorithm='racos', budget=budget, intermediate_result=True,
                              intermediate_freq=intermediate_freq, seed=seed,
                              parallel=True, server_num=server_num, init_samples=init_samples)
        parameter.set_probability(0.6)
        solution_list = ExpOpt.min(objective, parameter, repeat=repeat, plot=plot, plot_file=plot_file)

        f_min = np.inf
        x_min = None
        for s in solution_list:
            if s.get_value() < f_min:
                f_min = s.get_value()
                x_min = s.get_x()

        self.set_param(x_min)

        return x_min, f_min

    def get_dim(self):
        return self.dim, self.dim_range, self.dim_type

    def set_param(self, x):
        assert len(x) == self.dim, 'Parameter Number Error!'

        x_new = [min(xx, uu) for (xx, uu) in zip(x, self.upper)]
        x_new = [max(xx, ll) for (xx, ll) in zip(x_new, self.lower)]
        if x_new != x:
            log.warning('The parameter value has been changed!')

        log.info(f"Setting params: {x_new}")
        for idx, m in enumerate(self.models):
            m.set_param(x[self.dims[idx]: self.dims[idx + 1]])
        self.param = x_new


    def simulate(self, start_date, duration, round_result=True, simulate_date=None,
                 simulate_touchratio=None, population_ratio=1, simulate_isoratio=None,
                 simulate_population_ratio=1, simulate_out_ratio=1):
        city = self.city
        # population_ratio = 0
        # initialization
        result_sim = {}
        history = []

        date, sim_date_list = gen_date_list(start_date, duration)
        unob_period = self.unob_period
        init_unob_pre = unob_period * [0]
        if self.unob_flow_num is not None:
            date_it = date
            while True:
                if date_it not in self.unob_flow_num:
                    break
                interval = (date - date_it).days + 1
                init_unob_pre[-interval] = self.unob_flow_num[date_it][city] * population_ratio * (1 + self.infectratio.predict(city, date)
                                  * self.touchratiointra.predict(city, date)[0])
                date_it = date_it - timedelta(1)
        infection = {}

        infection[city] = InfectionList(self.dummy.predict()[0],
                                        self.infectratio.predict(city, date), self.obs.predict(city, date),
                                        self.isoratio.predict(city, date), incubation=self.incubation,
                                        population=self.population, init_unob_pre=init_unob_pre,
                                        base_touch_num=self.base_touch_num, unob_period=self.unob_period,
                                        obs_period=self.obs_period, iso_period=self.iso_period,
                                        cure_period=self.cure_period, no_healed=self.no_healed)
        inc_incity = {}
        inc_incity[city] = 0
        date_result = gen_date_result(city, date, infection, inc_incity, round_result)

        result_sim[date_result['date']] = date_result['record']
        predict_records = gen_history(city, date, infection, round_result)
        history += predict_records
        last_num = 0
        last_flow_out = 0
        # simulation
        for date in sim_date_list:
            infect_ratio = self.infectratio.predict(city, date) * self.base_touch_num
            for i in range(len(self.important_dates)):
                if date >= self.important_dates[i]:
                    infect_ratio = self.infectratio.predict(city, date) * self.touchratio.predict(city, date)[i] * self.base_touch_num
            if simulate_date is not None and date >= simulate_date:
                infect_ratio = self.infectratio.predict(city, date) * simulate_touchratio * self.base_touch_num
                if simulate_isoratio is not None:
                    infection[city].isoratio = simulate_isoratio
                population_ratio = simulate_population_ratio
                infection[city].out_ratio = simulate_out_ratio
            delta_infection_incity = infection[city].total_unob_infection * infect_ratio
            # update unobserved cured / deceased
            new_infection = delta_infection_incity
            if self.unob_flow_num is not None and date in self.unob_flow_num:
                flow_in_num = self.unob_flow_num[date][city]
                last_num = flow_in_num
            else:
                flow_in_num = last_num
                flow_in_num = 0
            #last_num = flow_out_num
            #intra_new_infection = flow_in_num * population_ratio * (1 + self.infectratio.predict(city, date)
            #                      * self.touchratiointra.predict(city, date)[0])


            intra_new_infection = flow_in_num * population_ratio * (1 + self.infectratio.predict(city, date) * self.base_touch_num
                                                                    * self.touchratiointra.predict(city, date)[0])
            dead_rate_ob = self.dead.predict()[0]
            dead_rate_unob = 0
            if self.flow_out_data is not None:
                if date in self.flow_out_data:
                    flow_out_num = self.flow_out_data[date][city]
                    last_flow_out = flow_out_num
                else:
                    flow_out_num = last_flow_out
            else:
                flow_out_num = 0

            popu_r = (self.population - infection[city].cum_infection) / self.population
            infection[city].update(popu_r * new_infection, intra_new_infection, dead_rate_ob, dead_rate_unob, self.cure_ratio.predict(date, city), flow_out_num)
            inc_incity[city] = delta_infection_incity
            date_result = gen_date_result(city, date, infection, inc_incity, round_result)
            result_sim[date_result['date']] = date_result['record']
            predict_records = gen_history(city, date, infection, round_result)
            history += predict_records
        self.history = pd.DataFrame(history)
        return self.history, infection[city]
