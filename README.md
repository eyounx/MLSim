# MLSim
MLSim is a machine learning based fine-grained disease transmission simulator. The algorithm is based on the paper "COVID-19 Asymptomatic Infection Estimation". 

To forecast the epidemic progress, MLSim integrates multiple practical factors including disease progress in the incubation period, cross-region population movement, undetected asymptomatic patients, and prevention and containment strength. The interactions among these factors are modeled by virtual transmission dynamics with several undetermined parameters, which are determined from epidemic data by machine learning techniques. 

A trained MLSim model can forecast the epidemic progress (including the infected, confirmed, mortalled, recovery, quarantined, self-healed cases each day) under different setting, e.g., to forecast what if the containment was postponed, weakened or strengthened in China or other countries.

## Installation
MLSim can be run either locally or through Docker.

### Local installation requirement
- python3.6
- pytorch
- psutil
- ZOOpt

To install ZOOpt, the following commands are recommended.

```bash
git clone -b multiprocess https://github.com/eyounx/ZOOpt.git
cd ZOOpt
python setup.py build
python setup.py install
```
### Docker
We also provided a Dockerfile to help install the dependencies.

```bash
docker build -t ncov .
docker run -it --rm --name python -v $PWD:/root/ncov ncov
cd ncov
```

### Data source and data format
We released our core algorithm. However, because of the copyright reasons, we didn't release all data. Before using MLSim to train your own model, you should gather the corresponding data yourself.

The Chinese epidemic data was gathered from [BlankerL/DXY-COVID-19-Data](https://github.com/BlankerL/DXY-COVID-19-Data).
The epidemic data source of other countries is [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19).
The population movement data in China’s mainland were sourced from [baiduqianxi](http://qianxi.baidu.com).
Here are the data descriptions.

- `data/data_x-xx-foreign.csv` epidemic data in other countries, Without this file, the experiments related to other countries can't be conducted.
- `data/province_data_extra_x-xx.csv` epidemic data in China's mainland. Without this file, the experiments related to China's mainland can't be conducted.
- `data/refine_transfer3.csv` population movement data in China's mainland. Without this file, the experiments related to China's mainland except for Hubei can't be conducted.
- `data/cur_confirmed-xxxx-xx-xx.json` cumulative confirmed cases in China's mainland on the date of xxxx-xx-xx. This file is only used in the results plot function, i.e., without this file, forecasting and training will not be influenced.

The epidemic data table should contain the following columns.

- `date` current date in the format of `yyyy-mm-dd`.
- `adcode` address code of current region/province. Region mapping relationship is listed in the function of `it_code_dict` in `components/utils.py`.
- `dead` newly mortalled cases.
- `cum_dead` cumulative mortalled cases.
- `cured` newly recovered cases.
- `cum_cured`cumulative recovered cases.
- `observed` newly confirmed cases.
- `cum_confirmed` cumulative confirmed cases.

The population movement data table should contain following columns.

- `date` current date in the format of `yyyy-mm-dd`.
- `src_id` address code of the source province.
- `dst_id` address code of the destination province.
- `travel_num` population of the movement from source to destination.



## Run
If all data is gathered, following instructions will lead you to train the model and forecast under different settings.


### Epidemic progress in other countries

```bash
python run_foreign.py					# train the parameters
python prepare_data_foreign.py		# forcast under different settings, analysis results
python paper_plot2_foreign.py			# plot the figures in papers
```
 
### Epidemic progress in China's mainland

```bash
python run.py                    	# train the parameters
python prepare_data_china.py		# forcast under different settings, analysis results
python paper_plot2_china.py		# plot the figures in papers
```


### Comprasion with SEIR and LSTM methods

```bash
python comparison.py
```
By default, `comparison.py` forecasts the results in China's mainland, change the `korea_flag` in line 22 to True to forecast the results in South Korea.

## Analysis and plot the pre-generated data
We can't release all of our data for the copyright reasons. But we released the trained model and pre-generared data in `trained_model` directory. Following the below instructions to reproduce our results.

Copy the pre-generated data into the current folder.

```bash
find ./trained_model/ -type f | xargs -I {} cp {} .
```


### Epidemic progress in other countries

```bash
python prepare_data_foreign.py --load_inter 1
python paper_plot2_foreign.py
```
If you want to change the forecasting configuration in these countries, refer to the function `run_simulation2` in `show_simulation_foreign.py` for more details. And you can forecast the results by the following commands.

```bash
python prepare_data_foreign.py 
python paper_plot2_foreign.py
```

### Epidemic progress in China's mainland

```bash
python prepare_data_china.py --load_inter 1
python papar_plot2_china.py
```
For lack of the population movement data, a new forecasting configuration can not be set.


### Comprasion with SEIR and LSTM methods

```bash
python comparison.py --load_inter 1
```
