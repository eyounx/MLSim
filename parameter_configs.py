from components.models import ObservationRatio, InfectRatio, TouchRatio, DeadRatio, DummyModel, IsolationRatio

interval = 3
def hubei():
    infectratio = InfectRatio(1, [[0, 0.1]], [True])
    touchratio = TouchRatio(1, [[0.0, 1/3.]], [True])  # 0.33333  0.2
    touchratiointra = TouchRatio(1, [[0., 1.]], [True])
    obs = ObservationRatio(1, [[0.0, 0.3]], [True])  # 0.2 0.3
    dead = DeadRatio(1, [[0., 0.01]], [True])
    isoratio = IsolationRatio(1, [[0.03, 0.24]], [True])  # 0.03 0.12  0.24
    dummy = DummyModel(1, [[0, 400]], [True, True])
    cure_ratio = InfectRatio(1, [[0., 0.1]], [True])
    budget = 200000
    return 'Hubei', [infectratio.dim_range[0], [round((item * 15), interval) for item in touchratio.dim_range[0]], obs.dim_range[0],
            dead.dim_range[0], dummy.dim_range[0], isoratio.dim_range[0], [round(item * 15, interval) for item in touchratiointra.dim_range[0]], cure_ratio.dim_range[0],
            budget]

def china():
    infectratio = InfectRatio(1, [[0, 0.05]], [True])
    touchratio = TouchRatio(1, [[0.0, 1/3.]], [True])
    touchratiointra = TouchRatio(1, [[0., 1.]], [True])  # heilongjiang and
    obs = ObservationRatio(1, [[0.0, 0.3]], [True])  # 0.2 0.3
    dead = DeadRatio(1, [[0., 0.01]], [True])
    isoratio = IsolationRatio(1, [[0.03, 0.12]], [True])  # 0.03 0.12  0.24
    dummy = DummyModel(1, [[0, 0.00001]], [True, True])
    cure_ratio = InfectRatio(1, [[0., 0.1]], [True])
    budget = 80000
    return 'China (exclude Hubei)', [infectratio.dim_range[0], [round(item * 15, interval) for item in touchratio.dim_range[0]], obs.dim_range[0],
                     dead.dim_range[0], dummy.dim_range[0], isoratio.dim_range[0], [round(item * 15, interval) for item in touchratiointra.dim_range[0]], cure_ratio.dim_range[0],
                     budget]

def foreign1():
    infectratio = InfectRatio(1, [[0.015, 0.023]], [True])
    touchratio = TouchRatio(1, [[0, 0.4]], [True])
    touchratiointra = TouchRatio(1, [[0, 1]], [True])
    obs = ObservationRatio(1, [[0.0, 0.3]], [True])
    dead = DeadRatio(1, [[0., 0.01]], [True])
    isoratio = IsolationRatio(1, [[0.02, 0.12]], [True])
    dummy = DummyModel(1, [[0, 400]], [True, True])
    cure_ratio = InfectRatio(1, [[0., 0.15]], [True])
    budget = 80000
    return 'Other country1', [infectratio.dim_range[0], [round(item * 15, interval) for item in touchratio.dim_range[0]], obs.dim_range[0],
                     dead.dim_range[0], dummy.dim_range[0], isoratio.dim_range[0], [round(item * 15, interval) for item in touchratiointra.dim_range[0]], cure_ratio.dim_range[0],
                     budget]

def foreign2():
    infectratio = InfectRatio(1, [[0.015, 0.023]], [True])
    touchratio = TouchRatio(1, [[0.999999, 0.999999999]], [True])
    touchratiointra = TouchRatio(1, [[0, 1]], [True])
    obs = ObservationRatio(1, [[0.0, 0.3]], [True])
    dead = DeadRatio(1, [[0., 0.01]], [True])
    isoratio = IsolationRatio(1, [[0.02, 0.12]], [True])
    dummy = DummyModel(1, [[0, 400]], [True, True])
    cure_ratio = InfectRatio(1, [[0., 0.15]], [True])
    budget = 80000
    return 'Other country2', [infectratio.dim_range[0], [round(item * 15, interval) for item in touchratio.dim_range[0]], obs.dim_range[0],
                     dead.dim_range[0], dummy.dim_range[0], isoratio.dim_range[0], [round(item * 15, interval) for item in touchratiointra.dim_range[0]], cure_ratio.dim_range[0],
                     budget]

def lstm_hubei():
    infectratio = InfectRatio(1, [[0, 0.1]], [True])
    touchratio = TouchRatio(1, [[0.0, 0.3333333]], [True])  # 0.33333  0.2
    touchratiointra = TouchRatio(1, [[0., 1.]], [True])
    obs = ObservationRatio(1, [[0.0, 0.3]], [True])  # 0.2 0.3
    dead = DeadRatio(1, [[0., 0.01]], [True])
    isoratio = IsolationRatio(1, [[0.03, 0.12]], [True])  # 0.03 0.12  0.24
    dummy = DummyModel(1, [[0, 400]], [True, True])
    cure_ratio = InfectRatio(1, [[0., 0.1]], [True])
    budget = 200000
    return 'Comparision Hubei', [infectratio.dim_range[0], [round(item * 15, interval) for item in touchratio.dim_range[0]], obs.dim_range[0],
                     dead.dim_range[0], dummy.dim_range[0], isoratio.dim_range[0], [round(item * 15, interval) for item in touchratiointra.dim_range[0]], cure_ratio.dim_range[0],
                     budget]

def lstm_china():
    infectratio = InfectRatio(1, [[0, 0.1]], [True])
    touchratio = TouchRatio(1, [[0.0, 0.3333333]], [True])  # 0.33333  0.2
    touchratiointra = TouchRatio(1, [[0., 1.]], [True])
    obs = ObservationRatio(1, [[0.0, 0.3]], [True])  # 0.2 0.3
    dead = DeadRatio(1, [[0., 0.01]], [True])
    isoratio = IsolationRatio(1, [[0.03, 0.12]], [True])  # 0.03 0.12  0.24
    dummy = DummyModel(1, [[0, 0.00001]], [True, True])
    cure_ratio = InfectRatio(1, [[0., 0.1]], [True])
    budget = 80000
    return 'Comparision China (exclude Hubei)', [infectratio.dim_range[0], [round(item * 15, interval) for item in touchratio.dim_range[0]], obs.dim_range[0],
                     dead.dim_range[0], dummy.dim_range[0], isoratio.dim_range[0], [round(item * 15, interval) for item in touchratiointra.dim_range[0]], cure_ratio.dim_range[0],
                     budget]



def lstm_foreign():
    infectratio = InfectRatio(1, [[0.015, 0.023]], [True])
    touchratio = TouchRatio(1, [[0, 0.4]], [True])
    touchratiointra = TouchRatio(1, [[0, 1]], [True])
    obs = ObservationRatio(1, [[0.0, 0.3]], [True])
    dead = DeadRatio(1, [[0., 0.01]], [True])
    isoratio = IsolationRatio(1, [[0.02, 0.12]], [True])
    dummy = DummyModel(1, [[0, 400]], [True, True])
    cure_ratio = InfectRatio(1, [[0., 0.15]], [True])
    budget = 80000

    return 'Comparision other country', [infectratio.dim_range[0], [round(item * 15, interval) for item in touchratio.dim_range[0]], obs.dim_range[0],
                     dead.dim_range[0], dummy.dim_range[0], isoratio.dim_range[0], [round(item * 15, interval) for item in touchratiointra.dim_range[0]], cure_ratio.dim_range[0],
                     budget]


def main():
    keys = []
    ranges = []
    func = [lstm_hubei, lstm_china, lstm_foreign,hubei, china, foreign1, foreign2, ]
    labels = ['$\\beta$', '$k$', '$\\gamma$', '$\\delta$', '$I(0)$', '$\\sigma$', '$k\'$', '$\\theta$', 'budget']
    for item in func:
        key, r = item()
        keys.append(key)
        ranges.append(r)
    print('&',sep='', end='')
    for item in labels:
        print('&{}'.format(item), sep='', end='')
    print('\\\\ \\hline')
    for ind, item in enumerate(ranges):
        print(' & ', sep='', end='')
        print(keys[ind], ' ', sep='', end='')
        for item1 in item:
            if isinstance(item1, list):
                print('& [{}, {}]'.format(item1[0], item1[1]), sep='', end='')
            else:
                print('& {}'.format(item1), sep='',end='')
        print('\\\\')
    # heilongjiang and

    # china






if __name__ == '__main__':
    main()
