from datetime import date, timedelta, datetime
import json, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../ncovmodel"))
from components.models import ObservationRatio, InfectRatio, TouchRatio, DummyModel, DeadRatio, IsolationRatio
import pandas as pd
import numpy as np
from components.utils import prepareData, flowOutData, flowHubei, get_data_path, append_to_json, get_important_date, get_R0, get_DT, get_death_rate
from components.simulator import Simulator, get_loss, get_newly_loss
import matplotlib
font = {'size'   : 18}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.pyplot import MultipleLocator
from datetime import date, timedelta
import matplotlib.ticker as ticker

minimal_r = 1.6
middle_r = 1.9
maximum_r = 2.1

start_date = date(2020, 1, 11)
total_model = False

def plot1(real, simulated, title, savefile, interval=7):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    xs = [start_date + timedelta(days=i) for i in range(len(real))]
    plt.plot(xs, real, color="indianred", linewidth=2.5, linestyle="-", label="real")
    xs = [start_date + timedelta(days=i) for i in range(len(simulated))]
    plt.plot(xs,simulated, color="cornflowerblue", linewidth=2.5, linestyle="-",  label="MLSim")
    plt.gcf().autofmt_xdate()
    #plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Cases number', fontsize=18)
    ax=plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    #ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim(bottom=0)
    if total_model:
        plt.title(title)
        plt.savefig(savefile[:-4]+'_nolabel.pdf', bbox_inches='tight')
        plt.clf()
        return
    plt.legend(loc='upper left')
    #plt.title(title)
    plt.savefig(savefile, bbox_inches='tight')
    plt.clf()

def plot1_shade(real, simulated, smin, smax, title, savefile, interval=7):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    xs = [start_date + timedelta(days=i) for i in range(len(real))]
    plt.plot(xs, real, color="indianred", linewidth=2.5, linestyle="-", label="real")
    xs = [start_date + timedelta(days=i) for i in range(len(simulated))]
    plt.plot(xs,simulated, color="cornflowerblue", linewidth=2.5, linestyle="-",  label="MLSim")
    plt.fill_between(xs, smin, smax, color='cornflowerblue', alpha=0.25)
    plt.gcf().autofmt_xdate()
    #plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Cases number', fontsize=18)
    ax=plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    #ax.xaxis.set_major_locator(x_major_locator)
    plt.ylim(bottom=0)
    if total_model:
        plt.title(title)
        plt.savefig(savefile[:-4]+'_nolabel.pdf', bbox_inches='tight')
        plt.clf()
        return
    plt.legend(loc='upper left')
    #plt.title(title)
    plt.savefig(savefile, bbox_inches='tight')
    plt.clf()

def plot11( simulated, title, savefile):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    xs = [start_date + timedelta(days=i) for i in range(len(simulated))]
    plt.plot(xs,simulated, color="cornflowerblue", linewidth=2.5, linestyle="-",  label="total infection")
    plt.gcf().autofmt_xdate()
    #plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Cases number', fontsize=18)
    ax=plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
    #ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(loc='upper left')
    plt.ylim(bottom=0)
    #plt.title(title)
    plt.savefig(savefile, bbox_inches='tight')
    plt.clf()

def plot12(real, simulated, newly_infection, title, savefile):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    xs = [start_date + timedelta(days=i) for i in range(len(real))]
    plt.plot(xs, real, color="indianred", linewidth=2.5, linestyle="-", label="real-confirmed")
    xs = [start_date + timedelta(days=i) for i in range(len(simulated))]
    plt.plot(xs,simulated, color="cornflowerblue", linewidth=2.5, linestyle="-",  label="MLSim-confirmed")
    xs = [start_date + timedelta(days=i) for i in range(len(newly_infection))]
    plt.plot(xs,newly_infection, color="orange", linewidth=2.5, linestyle="-",  label="MLSim-infected")
    plt.gcf().autofmt_xdate()
    #plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Cases number', fontsize=18)
    ax=plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
    #ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(loc='best')
    plt.ylim(bottom=0)
    #plt.title(title)
    plt.savefig(savefile, bbox_inches='tight')
    plt.clf()

def plot12_shade(real, simulated, s_min, s_max, newly_infection, ni_min, ni_max, title, savefile):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    xs = [start_date + timedelta(days=i) for i in range(len(real))]
    plt.plot(xs, real, color="indianred", linewidth=2.5, linestyle="-", label="real-confirmed")
    xs = [start_date + timedelta(days=i) for i in range(len(simulated))]
    plt.plot(xs,simulated, color="cornflowerblue", linewidth=2.5, linestyle="-",  label="MLSim-confirmed")
    plt.fill_between(xs, s_min, s_max, color='cornflowerblue', alpha=0.25)
    xs = [start_date + timedelta(days=i) for i in range(len(newly_infection))]
    plt.plot(xs,newly_infection, color="orange", linewidth=2.5, linestyle="-",  label="MLSim-infected")
    plt.fill_between(xs, ni_min, ni_max, color='orange', alpha=0.25)
    plt.gcf().autofmt_xdate()
    #plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Cases number', fontsize=18)
    ax=plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
    #ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(loc='best')
    plt.ylim(bottom=0)
    #plt.title(title)
    plt.savefig(savefile, bbox_inches='tight')
    plt.clf()

def plot33_shade(s1, s1_min, s1_max,
                 s2, s2_min, s2_max,
                 s3, s3_min, s3_max,
                 s4, s4_min, s4_max, history, title, savefile, touchratio, legend_size=7, loc='upper left',
           ratio_low=0.1, ratio_high=0.2, date_it=None, xmin=None, auto_ymin=True, yax_max=None,
           ext_flag=False):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    x_range_1 = np.array(range(len(s1)))
    x_range_2 = np.array(range(len(s3)))
    x_range_3 = np.array(range(len(s3)))
    x_range_his = np.array(range(len(history)))
    length = len(x_range_his) -1
    x_range_1 -= length
    x_range_2 -= length
    x_range_3 -= length
    x_range_his -= length
    if xmin is not None:
        range_s = (xmin-start_date).days
        start_date_it = xmin
        s1 = s1[range_s:]
        s2 = s2[range_s:]
        s3 = s3[range_s:]
        if s4 is not None:
            s4 = s4[range_s:]
        history = history[range_s:]
    else:
        start_date_it = start_date

    y_max = max(np.max(s1), np.max(s2), np.max(s3), np.max(history))
    y_min = min(np.min(s1), np.min(s2), np.min(s3), np.min(history))
    if s4 is not None:
        y_max = max(y_max, np.max(s4))
        y_min = min(y_min, np.min(s4))
    alphas = 0.25
    xs = [start_date_it + timedelta(days=i) for i in range(len(s1))]
    plt.plot(xs, s1, color="cornflowerblue", linewidth=2.5, linestyle="-", label="{}".format('100%'))
    plt.fill_between(xs, s1_min, s1_max, color='cornflowerblue', alpha=alphas)
    #plt.plot(xs, s1, color="cornflowerblue", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(round(touchratio, 5)))
    xs = [start_date_it + timedelta(days=i) for i in range(len(s2))]
    plt.plot(xs, s2, color="orange", linewidth=2.5, linestyle="-", label="{:.0f}%".format(middle_r*100))
    plt.fill_between(xs, s2_min, s2_max, color='orange', alpha=alphas)
    #plt.plot(xs, s2, color="orange", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(ratio_high))
    xs = [start_date_it + timedelta(days=i) for i in range(len(s3))]
    #plt.plot(xs, s3, color="forestgreen", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(ratio_low))
    plt.plot(xs, s3, color="forestgreen", linewidth=2.5, linestyle="-", label="{:.0f}%".format(100 * minimal_r))
    plt.fill_between(xs, s3_min, s3_max, color='forestgreen', alpha=alphas)
    if s4 is not None:
        xs = [start_date_it + timedelta(days=i) for i in range(len(s4))]
        #plt.plot(xs, s3, color="forestgreen", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(ratio_low))
        if ext_flag:
            plt.plot(xs, s4, color="rosybrown", linewidth=2.5, linestyle="-", label="{:.0f}%".format(100 * 1.3))
            plt.fill_between(xs, s4_min, s4_max, color='rosybrown', alpha=alphas)
        else:
            plt.plot(xs, s4, color="purple", linewidth=2.5, linestyle="-", label="{:.0f}%".format(100 * maximum_r))
            plt.fill_between(xs, s4_min, s4_max, color='purple', alpha=alphas)
    xs = [start_date_it + timedelta(days=i) for i in range(len(history))]
    plt.plot(xs, history, color='indianred', linewidth=2.5, linestyle="-", label="real")
    if date_it is not None:
        if auto_ymin:
            plt.vlines(date_it, 0, y_max, colors="r", linestyles="dashed")
        else:
            plt.vlines(date_it, y_min, y_max, colors="r", linestyles="dashed")

    plt.gcf().autofmt_xdate()
    #plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Cases number', fontsize=18)
    #plt.title(title)
    #x_major_locator = MultipleLocator(6)
    ax = plt.gca()
    if xmin  is None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    #ax.xaxis.set_major_locator(x_major_locator)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    if auto_ymin:
        plt.ylim(bottom=0)
    if yax_max is not None:
        plt.ylim(top=yax_max)

    if xmin is not None:
        plt.xlim(left=xmin)
    #ax.spines['left'].set_position(('data', 0))
    if total_model:
        plt.title(title)
        plt.savefig(savefile[:-4]+'_nolabel.pdf', bbox_inches='tight')
        plt.clf()
        return
    plt.legend(loc=loc)
    plt.savefig(savefile, bbox_inches='tight')
    plt.clf()

def plot33(s1, s2, s3, s4, history, title, savefile, touchratio, legend_size=7, loc='upper left',
           ratio_low=0.1, ratio_high=0.2, date_it=None, xmin=None, auto_ymin=True, yax_max=None,
           ext_flag=False):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    x_range_1 = np.array(range(len(s1)))
    x_range_2 = np.array(range(len(s3)))
    x_range_3 = np.array(range(len(s3)))
    x_range_his = np.array(range(len(history)))
    length = len(x_range_his) -1
    x_range_1 -= length
    x_range_2 -= length
    x_range_3 -= length
    x_range_his -= length
    if xmin is not None:
        range_s = (xmin-start_date).days
        start_date_it = xmin
        s1 = s1[range_s:]
        s2 = s2[range_s:]
        s3 = s3[range_s:]
        if s4 is not None:
            s4 = s4[range_s:]
        history = history[range_s:]
    else:
        start_date_it = start_date

    y_max = max(np.max(s1), np.max(s2), np.max(s3), np.max(history))
    y_min = min(np.min(s1), np.min(s2), np.min(s3), np.min(history))
    if s4 is not None:
        y_max = max(y_max, np.max(s4))
        y_min = min(y_min, np.min(s4))

    xs = [start_date_it + timedelta(days=i) for i in range(len(s1))]
    plt.plot(xs, s1, color="cornflowerblue", linewidth=2.5, linestyle="-", label="{}".format('100%'))
    #plt.plot(xs, s1, color="cornflowerblue", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(round(touchratio, 5)))
    xs = [start_date_it + timedelta(days=i) for i in range(len(s2))]
    plt.plot(xs, s2, color="orange", linewidth=2.5, linestyle="-", label="{:.0f}%".format(middle_r*100))
    #plt.plot(xs, s2, color="orange", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(ratio_high))
    xs = [start_date_it + timedelta(days=i) for i in range(len(s3))]
    #plt.plot(xs, s3, color="forestgreen", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(ratio_low))
    plt.plot(xs, s3, color="forestgreen", linewidth=2.5, linestyle="-", label="{:.0f}%".format(100 * minimal_r))
    if s4 is not None:
        xs = [start_date_it + timedelta(days=i) for i in range(len(s4))]
        #plt.plot(xs, s3, color="forestgreen", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(ratio_low))
        if ext_flag:
            plt.plot(xs, s4, color="rosybrown", linewidth=2.5, linestyle="-", label="{:.0f}%".format(100 * 1.3))
        else:
            plt.plot(xs, s4, color="purple", linewidth=2.5, linestyle="-", label="{:.0f}%".format(100 * maximum_r))
    xs = [start_date_it + timedelta(days=i) for i in range(len(history))]
    plt.plot(xs, history, color='indianred', linewidth=2.5, linestyle="-", label="real")
    if date_it is not None:
        if auto_ymin:
            plt.vlines(date_it, 0, y_max, colors="r", linestyles="dashed")
        else:
            plt.vlines(date_it, y_min, y_max, colors="r", linestyles="dashed")

    plt.gcf().autofmt_xdate()
    #plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Cases number', fontsize=18)
    #plt.title(title)
    #x_major_locator = MultipleLocator(6)
    ax = plt.gca()
    if xmin  is None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    #ax.xaxis.set_major_locator(x_major_locator)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    if auto_ymin:
        plt.ylim(bottom=0)
    if yax_max is not None:
        plt.ylim(top=yax_max)

    if xmin is not None:
        plt.xlim(left=xmin)
    #ax.spines['left'].set_position(('data', 0))
    if total_model:
        plt.title(title)
        plt.savefig(savefile[:-4]+'_nolabel.pdf', bbox_inches='tight')
        plt.clf()
        return
    plt.legend(loc=loc)
    plt.savefig(savefile, bbox_inches='tight')
    plt.clf()

def plot33_time(s1, s2, s3, s4, s5, history, title, savefile, touchratio, legend_size=7, loc='upper left',
           ratio_low=0.1, ratio_high=0.2, date_it=None, xmin=None, auto_ymin=True, yax_max=None,
           ext_flag=False):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    x_range_1 = np.array(range(len(s1)))
    x_range_2 = np.array(range(len(s3)))
    x_range_3 = np.array(range(len(s3)))
    x_range_his = np.array(range(len(history)))
    length = len(x_range_his) -1
    x_range_1 -= length
    x_range_2 -= length
    x_range_3 -= length
    x_range_his -= length
    if xmin is not None:
        range_s = (xmin-start_date).days
        start_date_it = xmin
        s1 = s1[range_s:]
        s2 = s2[range_s:]
        s3 = s3[range_s:]
        if s4 is not None:
            s4 = s4[range_s:]
        history = history[range_s:]
    else:
        start_date_it = start_date

    y_max = max(np.max(s1), np.max(s2), np.max(s3), np.max(history))
    y_min = min(np.min(s1), np.min(s2), np.min(s3), np.min(history))
    if s4 is not None:
        y_max = max(y_max, np.max(s4))
        y_min = min(y_min, np.min(s4))

    xs = [start_date_it + timedelta(days=i) for i in range(len(s1))]
    plt.plot(xs, s1, color="cornflowerblue", linewidth=2.5, linestyle="-", label="{}".format('Jan. 23'))
    #plt.plot(xs, s1, color="cornflowerblue", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(round(touchratio, 5)))
    xs = [start_date_it + timedelta(days=i) for i in range(len(s2))]
    plt.plot(xs, s2, color="orange", linewidth=2.5, linestyle="-", label="{}".format('Jan. 24'))
    #plt.plot(xs, s2, color="orange", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(ratio_high))
    xs = [start_date_it + timedelta(days=i) for i in range(len(s3))]
    #plt.plot(xs, s3, color="forestgreen", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(ratio_low))
    plt.plot(xs, s3, color="forestgreen", linewidth=2.5, linestyle="-", label="{}".format('Jan. 26'))
    xs = [start_date_it + timedelta(days=i) for i in range(len(s4))]
    plt.plot(xs, s4, color="purple", linewidth=2.5, linestyle="-", label="{}".format('Jan. 28'))
    xs = [start_date_it + timedelta(days=i) for i in range(len(s5))]
    plt.plot(xs, s5, color="rosybrown", linewidth=2.5, linestyle="-", label="{}".format('Jan. 30'))
    xs = [start_date_it + timedelta(days=i) for i in range(len(history))]
    plt.plot(xs, history, color='indianred', linewidth=2.5, linestyle="-", label="real")
    if date_it is not None:
        if auto_ymin:
            plt.vlines(date_it, 0, y_max, colors="r", linestyles="dashed")
        else:
            plt.vlines(date_it, y_min, y_max, colors="r", linestyles="dashed")

    plt.gcf().autofmt_xdate()
    #plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Cases number', fontsize=18)
    #plt.title(title)
    #x_major_locator = MultipleLocator(6)
    ax = plt.gca()
    if xmin  is None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    #ax.xaxis.set_major_locator(x_major_locator)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    if auto_ymin:
        plt.ylim(bottom=0)
    if yax_max is not None:
        plt.ylim(top=yax_max)

    if xmin is not None:
        plt.xlim(left=xmin)
    #ax.spines['left'].set_position(('data', 0))
    if total_model:
        plt.title(title)
        plt.savefig(savefile[:-4]+'_nolabel.pdf', bbox_inches='tight')
        plt.clf()
        return
    plt.legend(loc=loc)
    plt.savefig(savefile, bbox_inches='tight')
    plt.clf()

def plot33_time_shade(s1, s1_min, s1_max, s2, s2_min, s2_max,
                      s3, s3_min, s3_max, s4, s4_min, s4_max,
                      s5, s5_min, s5_max, history, title, savefile, touchratio, legend_size=7, loc='upper left',
           ratio_low=0.1, ratio_high=0.2, date_it=None, xmin=None, auto_ymin=True, yax_max=None,
           ext_flag=False):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    x_range_1 = np.array(range(len(s1)))
    x_range_2 = np.array(range(len(s3)))
    x_range_3 = np.array(range(len(s3)))
    x_range_his = np.array(range(len(history)))
    length = len(x_range_his) -1
    x_range_1 -= length
    x_range_2 -= length
    x_range_3 -= length
    x_range_his -= length
    if xmin is not None:
        range_s = (xmin-start_date).days
        start_date_it = xmin
        s1 = s1[range_s:]
        s1_min = s1_min[range_s:]
        s1_max = s1_max[range_s:]
        s2 = s2[range_s:]
        s2_min = s2_min[range_s:]
        s2_max = s2_max[range_s:]
        s3 = s3[range_s:]
        s3_min = s3_min[range_s:]
        s3_max = s3_max[range_s:]
        s5 = s5[range_s]
        s5_min = s5_min[range_s:]
        s5_max = s5_max[range_s:]
        if s4 is not None:
            s4 = s4[range_s:]
            s4_min = s4_min[range_s:]
            s4_max = s4_max[range_s:]
        history = history[range_s:]
    else:
        start_date_it = start_date

    y_max = max(np.max(s1), np.max(s2), np.max(s3), np.max(history))
    y_min = min(np.min(s1), np.min(s2), np.min(s3), np.min(history))
    if s4 is not None:
        y_max = max(y_max, np.max(s4))
        y_min = min(y_min, np.min(s4))
    alphas = 0.25
    xs = [start_date_it + timedelta(days=i) for i in range(len(s1))]
    plt.plot(xs, s1, color="cornflowerblue", linewidth=2.5, linestyle="-", label="{}".format('Jan. 23'))
    plt.fill_between(xs, s1_min, s1_max, color='cornflowerblue', alpha=alphas)
    #plt.plot(xs, s1, color="cornflowerblue", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(round(touchratio, 5)))
    xs = [start_date_it + timedelta(days=i) for i in range(len(s2))]
    plt.plot(xs, s2, color="orange", linewidth=2.5, linestyle="-", label="{}".format('Jan. 24'))
    plt.fill_between(xs, s2_min, s2_max, color='orange', alpha=alphas)
    #plt.plot(xs, s2, color="orange", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(ratio_high))
    xs = [start_date_it + timedelta(days=i) for i in range(len(s3))]
    #plt.plot(xs, s3, color="forestgreen", linewidth=2.5, linestyle="-", label="Touch ratio: {:.2f}".format(ratio_low))
    plt.plot(xs, s3, color="forestgreen", linewidth=2.5, linestyle="-", label="{}".format('Jan. 26'))
    plt.fill_between(xs, s3_min, s3_max, color='forestgreen', alpha=alphas)
    xs = [start_date_it + timedelta(days=i) for i in range(len(s4))]
    plt.plot(xs, s4, color="purple", linewidth=2.5, linestyle="-", label="{}".format('Jan. 28'))
    plt.fill_between(xs, s4_min, s4_max, color='purple', alpha=alphas)
    xs = [start_date_it + timedelta(days=i) for i in range(len(s5))]
    plt.plot(xs, s5, color="rosybrown", linewidth=2.5, linestyle="-", label="{}".format('Jan. 30'))
    plt.fill_between(xs, s5_min, s5_max, color='rosybrown', alpha=alphas)
    xs = [start_date_it + timedelta(days=i) for i in range(len(history))]
    plt.plot(xs, history, color='indianred', linewidth=2.5, linestyle="-", label="real")
    if date_it is not None:
        if auto_ymin:
            plt.vlines(date_it, 0, y_max, colors="r", linestyles="dashed")
        else:
            plt.vlines(date_it, y_min, y_max, colors="r", linestyles="dashed")

    plt.gcf().autofmt_xdate()
    #plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Cases number', fontsize=18)
    #plt.title(title)
    #x_major_locator = MultipleLocator(6)
    ax = plt.gca()
    if xmin  is None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    #ax.xaxis.set_major_locator(x_major_locator)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    if auto_ymin:
        plt.ylim(bottom=0)
    if yax_max is not None:
        plt.ylim(top=yax_max)

    if xmin is not None:
        plt.xlim(left=xmin)
    #ax.spines['left'].set_position(('data', 0))
    if total_model:
        plt.title(title)
        plt.savefig(savefile[:-4]+'_nolabel.pdf', bbox_inches='tight')
        plt.clf()
        return
    plt.legend(loc=loc)
    plt.savefig(savefile, bbox_inches='tight')
    plt.clf()


def run_simulation2(x, city, simulate_day1, simulate_day2, start_date, simulate_date,
                     history_real,
                    incubation=3, unob_flow_num=None, json_name='data_run.json'):
    days_predict = 0

    infectratio = InfectRatio(1, [[0, 1]], [True])
    touchratio = TouchRatio(1, [[0, 1.]], [True])
    touchratiointra = TouchRatio(1, [[0, 10]], [True])
    obs = ObservationRatio(1, [[0.0, 1.]], [True])
    dead = DeadRatio(1, [[0., 0.1]], [True])
    isoratio = IsolationRatio(1, [[0., 1]], [True])
    dummy = DummyModel(1, [[0, 2000000]], [True, True])
    cure_ratio = InfectRatio(1, [[0., 100]], [True])

    flow_out_data = flowOutData()
    # set the time of applying touchratio
    important_dates = [get_important_date(city)]
    json_path = json_name
    end_date = datetime.strptime(history_real['date'].max(), '%Y-%m-%d').date()
    test_date = datetime.strptime(history_real['date'].max(), '%Y-%m-%d').date() - timedelta(days_predict)
    history_real = history_real[history_real['adcode'] == city]
    history_real = history_real[history_real['date'] >= str(start_date)]
    history_train = history_real[history_real['date'] <= str(test_date)]
    duration = len(history_train["date"].unique()) - 1

    simulator = Simulator(city, infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra, cure_ratio,
                          important_dates,
                          unob_flow_num=unob_flow_num, flow_out_data=flow_out_data, incubation=incubation)
    if not os.path.exists('./img/{}'.format(city)):
        os.makedirs('./img/{}'.format(city))
    if not os.path.exists('./result'.format(city)):
        os.makedirs('./result'.format(city))

    with open('./result/result{}.txt'.format(city), 'w') as f:
        print('infection_ratio: [0, 1], touch_ratio_in_province: [0, 0.5]', file=f)
        print('touch_ratio_between_province: [0, 1]', file=f)
        print('observe_ratio_each_day: [0.05613, 1]', file=f)
        print('dead_ratio_observe: [0, 0.1], dead_ratio_unob: [0, 0.1]', file=f)
        print('initial_infection_hubei: [8, 8.01].', file=f)
        print('isolation_ratio: [0, 0.1]', file=f)
        print('BEST RESUTL:', file=f)
        print('x = ', x, file=f)
        simulator.set_param(x)
        print('##################################', file=f)
        print('HISTORY REAL', file=f)
        print('Data form ' + str(start_date) + ' to ' + str(end_date) + 'are used for training', file=f)
        print(history_train[history_train['adcode'] == city], file=f)
        print('##################################', file=f)
        print('SIMULATED RESULTS', file=f)
        print('start_date: ', start_date, 'duration: ', duration, file=f)
        sim, records = simulator.simulate(str(start_date), duration)
        append_to_json(json_path, 'x', str(city), list(x))
        append_to_json(json_path, 'R01', str(city), get_R0(15, x[0], x[5], 14))
        append_to_json(json_path, 'R02', str(city), get_R0(15 * x[1], x[0], x[5], 14))
        append_to_json(json_path, 'DT1', str(city), get_DT(15, x[0], x[5]))
        append_to_json(json_path, 'DT2', str(city), get_DT(15 * x[1], x[0], x[5]))
        append_to_json(json_path, 'death_rate', str(city), get_death_rate(x[3], x[7], 14, 10))

        append_to_json(json_path, 'real_confirmed', str(city), list(history_train['observed']))
        append_to_json(json_path, 'real_cum_confirmed', str(city), list(history_train['cum_confirmed']))
        append_to_json(json_path, 'real_cum_dead', str(city), list(history_train['cum_dead']))
        append_to_json(json_path, 'sim_confirmed', str(city), list(sim['observed']))
        append_to_json(json_path, 'sim_new_infection', str(city), list(sim['new_infection']))
        append_to_json(json_path, 'sim_cum_confirmed', str(city), list(sim['cum_confirmed']))
        append_to_json(json_path, 'sim_total_infection', str(city), list(sim['total_infection']))
        append_to_json(json_path, 'sim_cum_self_cured', str(city), list(sim['cum_cured']))
        append_to_json(json_path, 'sim_cum_nosymbol', str(city), list(sim['cum_no_symbol']))
        append_to_json(json_path, 'loss', str(city), get_loss(sim, history_train))
        append_to_json(json_path, 'newly_confirmed_loss', str(city), get_newly_loss(sim, history_train))
        print(f'loss: {get_loss(sim, history_train)}', file=f)
        cum_real = history_train['cum_confirmed'].values
        simulated_data, _ = simulator.simulate(str(start_date), duration)
        cum_simulated = simulated_data['cum_confirmed'].values
        print(simulated_data[simulated_data['adcode'] == city], file=f)

        s1, _ = simulator.simulate(str(start_date), duration + simulate_day1)
        touch_ratio_high = x[1] * middle_r
        touch_ratio_low = x[1] * minimal_r
        touch_ratio_ext_low = x[1] * 1.3
        s2, _ = simulator.simulate(str(start_date), duration + simulate_day2, simulate_date=simulate_date,
                                   simulate_touchratio=touch_ratio_high)
        s3, _ = simulator.simulate(str(start_date), duration + simulate_day2, simulate_date=simulate_date,
                                   simulate_touchratio=touch_ratio_low)

        s4, _ = simulator.simulate(str(start_date), duration + simulate_day2, simulate_date=date(2020, 3, 1), simulate_touchratio=touch_ratio_low)
        s5, _ = simulator.simulate(str(start_date), duration + simulate_day2, simulate_date=date(2020, 3, 1), simulate_touchratio=touch_ratio_high)
        s6, _ = simulator.simulate(str(start_date), duration + simulate_day2, simulate_date=date(2020, 3, 1), simulate_touchratio=x[1] * maximum_r)
        s7, _ = simulator.simulate(str(start_date), duration + simulate_day2, simulate_date=simulate_date, simulate_touchratio=x[1] * maximum_r)
        s8, _ = simulator.simulate(str(start_date), duration + simulate_day2, simulate_date=simulate_date,
                                   simulate_touchratio=touch_ratio_ext_low)
        s8, _ = simulator.simulate(str(start_date), duration + simulate_day2, simulate_date=simulate_date,
                                   simulate_touchratio=touch_ratio_ext_low)

        simulator = Simulator(city, infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra,
                              cure_ratio,
                              [important_dates[0] + timedelta(days=1)],
                              unob_flow_num=unob_flow_num, flow_out_data=flow_out_data, incubation=incubation)
        s9, _ = simulator.simulate(str(start_date), duration + simulate_day1)
        simulator = Simulator(city, infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra,
                              cure_ratio,
                              [important_dates[0] + timedelta(days=3)],
                              unob_flow_num=unob_flow_num, flow_out_data=flow_out_data, incubation=incubation)
        s10, _ = simulator.simulate(str(start_date), duration + simulate_day1)
        simulator = Simulator(city, infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra,
                              cure_ratio,
                              [important_dates[0] + timedelta(days=5)],
                              unob_flow_num=unob_flow_num, flow_out_data=flow_out_data, incubation=incubation)
        s11, _ = simulator.simulate(str(start_date), duration + simulate_day1)
        simulator = Simulator(city, infectratio, touchratio, obs, dead, dummy, isoratio, touchratiointra,
                              cure_ratio,
                              [important_dates[0] + timedelta(days=7)],
                              unob_flow_num=unob_flow_num, flow_out_data=flow_out_data, incubation=incubation)
        s12, _ = simulator.simulate(str(start_date), duration + simulate_day1)
        c_con = list(s1['cum_confirmed'])[len(list(history_train['observed']))-1]
        c_inf = list(s1['cum_infection'])[len(list(history_train['observed']))-1]
        append_to_json(json_path, 'current_asym', str(city), c_con / c_inf)
        c_con = list(s1['cum_confirmed'])[-1]
        c_inf = list(s1['cum_infection'])[-1]
        append_to_json(json_path, 'final_asym', str(city), c_con / c_inf)
        append_to_json(json_path, 'sim_cum_confirmed_deduction_s1', str(city), list(s1['cum_confirmed']))
        append_to_json(json_path, 'sim_cum_confirmed_deduction_s2', str(city), list(s2['cum_confirmed']))
        append_to_json(json_path, 'sim_cum_confirmed_deduction_s3', str(city), list(s3['cum_confirmed']))
        append_to_json(json_path, 'sim_cum_confirmed_deduction_s4', str(city), list(s4['cum_confirmed']))
        append_to_json(json_path, 'sim_cum_confirmed_deduction_s5', str(city), list(s5['cum_confirmed']))
        append_to_json(json_path, 'sim_cum_confirmed_deduction_s6', str(city), list(s6['cum_confirmed']))
        append_to_json(json_path, 'sim_cum_confirmed_deduction_s7', str(city), list(s7['cum_confirmed']))
        append_to_json(json_path, 'sim_cum_confirmed_deduction_s8', str(city), list(s8['cum_confirmed']))
        append_to_json(json_path, 'sim_cum_confirmed_deduction_s9', str(city), list(s9['cum_confirmed']))
        append_to_json(json_path, 'sim_cum_confirmed_deduction_s10', str(city), list(s10['cum_confirmed']))
        append_to_json(json_path, 'sim_cum_confirmed_deduction_s11', str(city), list(s11['cum_confirmed']))
        append_to_json(json_path, 'sim_cum_confirmed_deduction_s12', str(city), list(s12['cum_confirmed']))
        append_to_json(json_path, 'sim_cum_dead_s1', str(city), list(s1['cum_dead']))
        append_to_json(json_path, 'sim_cum_dead_s2', str(city), list(s2['cum_dead']))
        append_to_json(json_path, 'sim_cum_dead_s3', str(city), list(s3['cum_dead']))
        append_to_json(json_path, 'sim_cum_dead_s4', str(city), list(s4['cum_dead']))
        append_to_json(json_path, 'sim_cum_dead_s5', str(city), list(s5['cum_dead']))
        append_to_json(json_path, 'sim_cum_dead_s6', str(city), list(s6['cum_dead']))
        append_to_json(json_path, 'sim_cum_dead_s7', str(city), list(s7['cum_dead']))
        append_to_json(json_path, 'sim_cum_dead_s8', str(city), list(s8['cum_dead']))
        append_to_json(json_path, 'sim_cum_dead_s9', str(city), list(s9['cum_dead']))
        append_to_json(json_path, 'sim_cum_dead_s10', str(city), list(s10['cum_dead']))
        append_to_json(json_path, 'sim_cum_dead_s11', str(city), list(s11['cum_dead']))
        append_to_json(json_path, 'sim_cum_dead_s12', str(city), list(s12['cum_dead']))
        append_to_json(json_path, 'sim_confirmed_deduction_s1', str(city), list(s1['observed']))
        append_to_json(json_path, 'sim_confirmed_deduction_s2', str(city), list(s2['observed']))
        append_to_json(json_path, 'sim_confirmed_deduction_s3', str(city), list(s3['observed']))
        append_to_json(json_path, 'sim_confirmed_deduction_s4', str(city), list(s4['observed']))
        append_to_json(json_path, 'sim_confirmed_deduction_s5', str(city), list(s5['observed']))
        append_to_json(json_path, 'sim_confirmed_deduction_s6', str(city), list(s6['observed']))
        append_to_json(json_path, 'sim_confirmed_deduction_s7', str(city), list(s7['observed']))
        append_to_json(json_path, 'sim_confirmed_deduction_s8', str(city), list(s8['observed']))
        append_to_json(json_path, 'sim_confirmed_deduction_s9', str(city), list(s9['observed']))
        append_to_json(json_path, 'sim_confirmed_deduction_s10', str(city), list(s10['observed']))
        append_to_json(json_path, 'sim_confirmed_deduction_s11', str(city), list(s11['observed']))
        append_to_json(json_path, 'sim_confirmed_deduction_s12', str(city), list(s12['observed']))
        append_to_json(json_path, 'sim_infection_deduction_s1', str(city), list(s1['total_infection']))
        append_to_json(json_path, 'sim_infection_deduction_s2', str(city), list(s2['total_infection']))
        append_to_json(json_path, 'sim_infection_deduction_s3', str(city), list(s3['total_infection']))
        append_to_json(json_path, 'sim_infection_deduction_s4', str(city), list(s4['total_infection']))
        append_to_json(json_path, 'sim_infection_deduction_s5', str(city), list(s5['total_infection']))
        append_to_json(json_path, 'sim_infection_deduction_s6', str(city), list(s6['total_infection']))
        append_to_json(json_path, 'sim_infection_deduction_s7', str(city), list(s7['total_infection']))
        append_to_json(json_path, 'sim_infection_deduction_s8', str(city), list(s8['total_infection']))
        append_to_json(json_path, 'sim_infection_deduction_s9', str(city), list(s9['total_infection']))
        append_to_json(json_path, 'sim_infection_deduction_s10', str(city), list(s10['total_infection']))
        append_to_json(json_path, 'sim_infection_deduction_s11', str(city), list(s11['total_infection']))
        append_to_json(json_path, 'sim_infection_deduction_s12', str(city), list(s12['total_infection']))
        append_to_json(json_path, 'sim_cum_infection_deduction_s1', str(city), list(s1['cum_infection']))
        append_to_json(json_path, 'sim_cum_infection_deduction_s2', str(city), list(s2['cum_infection']))
        append_to_json(json_path, 'sim_cum_infection_deduction_s3', str(city), list(s3['cum_infection']))
        append_to_json(json_path, 'sim_cum_infection_deduction_s4', str(city), list(s4['cum_infection']))
        append_to_json(json_path, 'sim_cum_infection_deduction_s5', str(city), list(s5['cum_infection']))
        append_to_json(json_path, 'sim_cum_infection_deduction_s6', str(city), list(s6['cum_infection']))
        append_to_json(json_path, 'sim_cum_infection_deduction_s7', str(city), list(s7['cum_infection']))
        append_to_json(json_path, 'sim_cum_infection_deduction_s8', str(city), list(s8['cum_infection']))
        append_to_json(json_path, 'sim_cum_infection_deduction_s9', str(city), list(s9['cum_infection']))
        append_to_json(json_path, 'sim_cum_infection_deduction_s10', str(city), list(s10['cum_infection']))
        append_to_json(json_path, 'sim_cum_infection_deduction_s11', str(city), list(s11['cum_infection']))
        append_to_json(json_path, 'sim_cum_infection_deduction_s12', str(city), list(s12['cum_infection']))

        append_to_json(json_path, 'sim_total_infection_deduction_s1', str(city), list(s1['total_unobserved']))
        append_to_json(json_path, 'sim_total_infection_deduction_s2', str(city), list(s2['total_unobserved']))
        append_to_json(json_path, 'sim_total_infection_deduction_s3', str(city), list(s3['total_unobserved']))
        append_to_json(json_path, 'sim_total_infection_deduction_s4', str(city), list(s4['total_unobserved']))
        append_to_json(json_path, 'sim_total_infection_deduction_s5', str(city), list(s5['total_unobserved']))
        append_to_json(json_path, 'sim_total_infection_deduction_s6', str(city), list(s6['total_unobserved']))
        append_to_json(json_path, 'sim_total_infection_deduction_s7', str(city), list(s7['total_unobserved']))
        append_to_json(json_path, 'sim_total_infection_deduction_s8', str(city), list(s8['total_unobserved']))
        append_to_json(json_path, 'sim_total_infection_deduction_s9', str(city), list(s9['total_unobserved']))
        append_to_json(json_path, 'sim_total_infection_deduction_s10', str(city), list(s10['total_unobserved']))
        append_to_json(json_path, 'sim_total_infection_deduction_s11', str(city), list(s11['total_unobserved']))
        append_to_json(json_path, 'sim_total_infection_deduction_s12', str(city), list(s12['total_unobserved']))
        append_to_json(json_path, 'sim_cum_self_cured_deduction_s1', str(city), list(s1['cum_self_cured']))
        append_to_json(json_path, 'sim_cum_self_cured_deduction_s2', str(city), list(s2['cum_self_cured']))
        append_to_json(json_path, 'sim_cum_self_cured_deduction_s3', str(city), list(s3['cum_self_cured']))
        append_to_json(json_path, 'sim_cum_self_cured_deduction_s4', str(city), list(s4['cum_self_cured']))
        append_to_json(json_path, 'sim_cum_self_cured_deduction_s5', str(city), list(s5['cum_self_cured']))
        append_to_json(json_path, 'sim_cum_self_cured_deduction_s6', str(city), list(s6['cum_self_cured']))
        append_to_json(json_path, 'sim_cum_self_cured_deduction_s7', str(city), list(s7['cum_self_cured']))
        append_to_json(json_path, 'sim_cum_self_cured_deduction_s8', str(city), list(s8['cum_self_cured']))
        append_to_json(json_path, 'sim_cum_self_cured_deduction_s9', str(city), list(s9['cum_self_cured']))
        append_to_json(json_path, 'sim_cum_self_cured_deduction_s10', str(city), list(s10['cum_self_cured']))
        append_to_json(json_path, 'sim_cum_self_cured_deduction_s11', str(city), list(s11['cum_self_cured']))
        append_to_json(json_path, 'sim_cum_self_cured_deduction_s12', str(city), list(s12['cum_self_cured']))

        append_to_json(json_path, 'sim_cum_nosymbol_deduction_s1', str(city), list(s1['cum_no_symbol']))
        append_to_json(json_path, 'sim_cum_nosymbol_deduction_s2', str(city), list(s2['cum_no_symbol']))
        append_to_json(json_path, 'sim_cum_nosymbol_deduction_s3', str(city), list(s3['cum_no_symbol']))
        append_to_json(json_path, 'sim_cum_nosymbol_deduction_s4', str(city), list(s4['cum_no_symbol']))
        append_to_json(json_path, 'sim_cum_nosymbol_deduction_s5', str(city), list(s5['cum_no_symbol']))
        append_to_json(json_path, 'sim_cum_nosymbol_deduction_s6', str(city), list(s6['cum_no_symbol']))
        append_to_json(json_path, 'sim_cum_nosymbol_deduction_s7', str(city), list(s7['cum_no_symbol']))
        append_to_json(json_path, 'sim_cum_nosymbol_deduction_s8', str(city), list(s8['cum_no_symbol']))
        append_to_json(json_path, 'sim_cum_nosymbol_deduction_s9', str(city), list(s9['cum_no_symbol']))
        append_to_json(json_path, 'sim_cum_nosymbol_deduction_s10', str(city), list(s10['cum_no_symbol']))
        append_to_json(json_path, 'sim_cum_nosymbol_deduction_s11', str(city), list(s11['cum_no_symbol']))
        append_to_json(json_path, 'sim_cum_nosymbol_deduction_s12', str(city), list(s12['cum_no_symbol']))
        append_to_json(json_path, 'sim_total_isolation_deduction_s1', str(city), list(s1['total_isolation']))
        append_to_json(json_path, 'sim_total_isolation_deduction_s2', str(city), list(s2['total_isolation']))
        append_to_json(json_path, 'sim_total_isolation_deduction_s3', str(city), list(s3['total_isolation']))
        append_to_json(json_path, 'sim_total_isolation_deduction_s4', str(city), list(s4['total_isolation']))
        append_to_json(json_path, 'sim_total_isolation_deduction_s5', str(city), list(s5['total_isolation']))
        append_to_json(json_path, 'sim_total_isolation_deduction_s6', str(city), list(s6['total_isolation']))
        append_to_json(json_path, 'sim_total_isolation_deduction_s7', str(city), list(s7['total_isolation']))
        append_to_json(json_path, 'sim_total_isolation_deduction_s8', str(city), list(s8['total_isolation']))
        append_to_json(json_path, 'sim_total_isolation_deduction_s9', str(city), list(s9['total_isolation']))
        append_to_json(json_path, 'sim_total_isolation_deduction_s10', str(city), list(s10['total_isolation']))
        append_to_json(json_path, 'sim_total_isolation_deduction_s11', str(city), list(s11['total_isolation']))
        append_to_json(json_path, 'sim_total_isolation_deduction_s12', str(city), list(s12['total_isolation']))


        append_to_json(json_path, 'touch_ratio_low', str(city), touch_ratio_low)
        append_to_json(json_path, 'touch_ratio_hight', str(city), touch_ratio_high)


def run_simulation(x, city, simulate_day1, simulate_day2, start_date, simulate_date, incubation=3, unob_flow_num=None,
                   json_name='data_run.json', fake_flag=False):
    real_data = pd.read_csv(get_data_path(fake_flag=fake_flag))
    history_real = prepareData(real_data)
    run_simulation2(x, city, simulate_day1, simulate_day2, start_date, simulate_date,
                    history_real, incubation, unob_flow_num, json_name=json_name)


