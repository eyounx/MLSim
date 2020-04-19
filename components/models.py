class BasicModel:
    def __init__(self, name='basic', ndim=0, dim_range=None, dim_type=None, params=None):
        self.name = name
        self.ndim = ndim
        self.dim_range = dim_range or []
        self.dim_type = dim_type or []
        self.params = params or [0]*self.ndim

    def get_param_info(self):
        return self.ndim, self.dim_range, self.dim_type

    def set_param(self, params):
        self.params = params


class InfectRatio(BasicModel):
    def __init__(self, ndim=1, dim_range=None, dim_type=None):
        dim_range = dim_range or [[0, 10]]
        dim_type = dim_type or [True]
        params = [1] * ndim

        super().__init__('InfectionInCity', ndim, dim_range, dim_type, params)

    def predict(self, city_id, date='0000-00-00'):
        r0_incity = self.params[0]
        return r0_incity


class TouchRatio(BasicModel):
    def __init__(self, ndim=1, dim_range=None, dim_type=None):
        dim_range = dim_range or [[0, 1]]
        dim_type = dim_type or [True]
        params = [1] * ndim

        super().__init__('InfectionInCity', ndim, dim_range, dim_type, params)

    def predict(self, city_id, date='0000-00-00'):
        r0_incity = self.params
        return r0_incity


class ObservationRatio(BasicModel):

    def __init__(self, ndim=1, dim_range=None, dim_type=None):
        dim_range = dim_range or [[0, 1]]
        dim_type = dim_type or [True]
        params = [0.1] * ndim

        super().__init__('ObservationRatio', ndim, dim_range, dim_type, params)

    def predict(self, city_id, date='0000-00-00'):
        obs_ratio = self.params[0]
        return obs_ratio


class IsolationRatio(BasicModel):
    def __init__(self, ndim=1, dim_range=None, dim_type=None):
        dim_range = dim_range or [[0, 1]]
        dim_type = dim_type or [True]
        params = [0.5] * ndim
        super().__init__('IsolationRatio', ndim, dim_range, dim_type, params)

    def predict(self, city_id, date='0000-00-00'):
        isolation_ratio = self.params[0]
        return isolation_ratio


class DeadRatio(BasicModel):

    def __init__(self, ndim=2, dim_range=None, dim_type=None):
        dim_range = dim_range or [[0, 1]] * ndim
        dim_type = dim_type or [True] * ndim
        params = [0.01, 0.01]

        super().__init__('DeceaseRatio', ndim, dim_range, dim_type, params)

    def predict(self, date='0000-00-00'):
        return self.params


class DummyModel(BasicModel):

    def __init__(self, dim=1, dim_range=None, dim_type=None):
        dim_range = dim_range or [[0, 10000]]
        dim_type = dim_type or [True]
        params = [0]*dim

        super().__init__('DummyModel', dim, dim_range, dim_type, params)

    def predict(self):
        return self.params
