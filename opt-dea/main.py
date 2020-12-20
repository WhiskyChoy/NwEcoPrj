# -*- coding: utf-8 -*-
import geatpy as ea  # import geatpy
from MyProblem import MyProblem  # 导入自定义问题接口
import pandas as pd
import numpy as np
import os
from common import data_prefix, result_prefix, output_prefix
from datetime import datetime, timedelta
from random import randint


def run(date, n, T, ep, c_max_dir, d_dir, s_dir, desc, df_c_max=None, df_d=None, df_s=None, MAXGEN=500, F=0.5, XOVR=0.7, logTras=0, verbose=False, drawing=0, trappedValue=0, maxTrappedCount=1000, save_mode='csv'):
    if not os.path.exists(data_prefix):
        os.mkdir(data_prefix)

    if not os.path.exists(result_prefix):
        os.mkdir(result_prefix)

    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    date_str = f'{date.year}-{date.month}-{date.day}'

    output_dir = f'date={date_str} n={n} T={T} ep={ep} desc={desc}.csv'
    print(output_dir)

    range_index = range(0, n)

    if df_c_max is None:
        c_max_csv = open(f'{data_prefix}/{c_max_dir}')
        df_c_max = pd.read_csv(c_max_csv, header=None, index_col=None).values

    if df_d is None:
        d_csv = open(f'{data_prefix}/{d_dir}')
        df_d = pd.read_csv(d_csv, header=None, index_col=None).values

    if df_s is None:
        s_csv = open(f'{data_prefix}/{s_dir}')
        df_s_csv = pd.read_csv(s_csv, header=None, index_col=0)
        str_date = f'{date.year}/{date.month}/{date.day}'
        s_arr = df_s_csv.loc[str_date].values
        df_s = {i: s_arr[i] for i in range_index}

    """================================实例化问题对象==========================="""
    problem = MyProblem(n, T, ep, df_c_max, df_d, df_s)  # 生成问题对象
    """==================================种群设置=============================="""
    Encoding = 'RI'  # 编码方式
    NIND = 100  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes,
                      problem.ranges, problem.borders)  # 创建区域描述器
    # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    population = ea.Population(Encoding, Field, NIND)
    """================================算法参数设置============================="""
    myAlgorithm = ea.soea_DE_rand_1_bin_templet(
        problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = MAXGEN  # 最大进化代数
    myAlgorithm.mutOper.F = F  # 差分进化中的参数F
    myAlgorithm.recOper.XOVR = XOVR  # 重组概率
    myAlgorithm.trappedValue = trappedValue  # “进化停滞”判断阈值
    # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
    myAlgorithm.maxTrappedCount = maxTrappedCount
    myAlgorithm.logTras = logTras  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = verbose  # 设置是否打印输出日志信息
    myAlgorithm.drawing = drawing  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """===========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    # BestIndi.save()  # 把最优个体的信息保存到文件中
    """==================================输出结果=============================="""
    print('Number of evaluation：%s' % myAlgorithm.evalsNum)
    print('Time used in seconds %s' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('Best objective value：%s' % BestIndi.ObjV[0][0])
        final_table = np.zeros((10, 10), dtype=int)
        count = 0
        for i in range(10):
            for j in range(10):
                if i != j:
                    final_table[i][j] = BestIndi.Phen[0, count]
                    count += 1
        if save_mode == 'csv':
            df = pd.DataFrame(final_table, columns=None, index=None)
            df.to_csv(f'{output_prefix}/{output_dir}',
                    header=False, index=False, mode='w')
        elif save_mode == 'json':
            f = open(f'{output_prefix}/{date_str}.json', 'w', encoding='utf-8')
            f.write(f'{final_table.tolist()}')
            f.close()
    else:
        print('No feasible solution!')
    return myAlgorithm


def get_dates(start_date: datetime, end_date: datetime):
    step = timedelta(days=1)
    total_steps = (end_date - start_date) // step + 1
    dates = [start_date + i * step for i in range(0, total_steps)]
    return dates


def test_0():
    n = 10
    T = 1000
    ep = 0.6
    c_max_dir = 'c_max.csv'
    d_dir = 'd.csv'
    s_dir = 's.csv'
    desc = 'default'

    date = datetime(2020, 4, 16)

    run(date, n, T, ep, c_max_dir, d_dir, s_dir, desc, save_mode='json')


def test_1():
    n = 10
    T_s = [100, 1000, 10000]
    ep_s = [0.5, 0.6, 0.7, 0.8, 0.9]

    c_max_dir = 'c_max.csv'
    d_dir = 'd.csv'
    s_dir = 's.csv'
    desc = 'default'

    start_date = datetime(2020, 2, 16)
    end_date = datetime(2020, 4, 16)

    dates = get_dates(start_date, end_date)

    for T in T_s:
        for ep in ep_s:
            for date in dates:
                run(date, n, T, ep, c_max_dir, d_dir, s_dir, desc)


def test_2():
    n = 10
    T = 1000
    ep = 0.75
    c_max_dir = 'c_max.csv'
    d_dir = 'd.csv'
    s_table_1 = [[randint(0, 100) for _ in range(0, n)] for _ in range(0, n)]
    s_table_2 = [[randint(0, 1000) for _ in range(0, n)] for _ in range(0, n)]
    s_table_3 = [[10 for _ in range(0, n)] for _ in range(0, n)]
    s_table_4 = [[100 for _ in range(0, n)] for _ in range(0, n)]
    s_table_5 = [[500 for _ in range(0, n)] for _ in range(0, n)]
    s_table_6 = [[1000 for _ in range(0, n)] for _ in range(0, n)]
    s_tables = [s_table_1, s_table_2, s_table_3,
                s_table_4, s_table_5, s_table_6]
    descs = ['randint100', 'randint1000',
             'same10', 'same100', 'same500', 'same1000']

    for s_table in s_tables:
        print(s_table)

    start_date = datetime(2020, 2, 16)
    end_date = datetime(2020, 4, 16)

    dates = get_dates(start_date, end_date)

    for date in dates:
        for i in range(0, 6):
            df_s = s_tables[i]
            desc = descs[i]
            run(date=date, n=n, T=T, ep=ep, c_max_dir=c_max_dir,
                d_dir=d_dir, s_dir=None, desc=desc, df_s=df_s)


def test_3():
    n = 10
    T = 100
    ep = 0.8

    c_max_dir = 'c_max.csv'
    d_dir = 'd.csv'
    s_dir = 's.csv'
    desc = 'draw_process'

    date = datetime(2020, 4, 16)
    drawing = 1

    run(date, n, T, ep, c_max_dir, d_dir, s_dir, desc, drawing=drawing)


def test_4():
    n = 10
    T_s = [500, 750, 1000]
    ep = 0.6

    c_max_dir = 'c_max.csv'
    d_dir = 'd.csv'
    s_dir = 's.csv'
    desc = 'draw_process'

    date = datetime(2020, 4, 16)
    drawing = 1

    for T in T_s:
        run(date, n, T, ep, c_max_dir, d_dir, s_dir, desc, drawing=drawing)


def test_5():
    n = 10
    T = 100
    ep = 0.8
    T_s = [100, 500, 1000]
    ep_s = [0.6, 0.7, 0.8, 0.9]

    c_max_dir = 'c_max.csv'
    d_dir = 'd.csv'
    s_dir = 's.csv'
    desc = 'draw_process'

    trappedValue = 10e-6
    maxTrappedCount = 10

    date = datetime(2020, 4, 16)

    count_gen = np.zeros((3, 4))

    for i in range(0, 3):
        T = T_s[i]
        for j in range(0, 4):
            ep = ep_s[j]
            alg = run(date, n, T, ep, c_max_dir, d_dir, s_dir, desc,
                      trappedValue=trappedValue, maxTrappedCount=maxTrappedCount)
            count_gen[i][j] = alg.currentGen

    print(count_gen)


def test_6():
    n = 10
    T = 1000
    ep = 0.75
    df_c_max_1 = [[20 for _ in range(0, n)] for _ in range(0, n)]
    df_c_max_2 = [[40 for _ in range(0, n)] for _ in range(0, n)]
    df_c_max_s = [df_c_max_1, df_c_max_2]

    d_dir = 'd.csv'
    s_dir = 's.csv'
    desc_s = ['c_max=20', 'c_max=40']

    date = datetime(2020, 3, 27)

    for i in range(0, 2):
        df_c_max = df_c_max_s[i]
        desc = desc_s[i]
        run(date, n, T, ep, c_max_dir=None, d_dir=d_dir,
            s_dir=s_dir, desc=desc, df_c_max=df_c_max)


def test_7():
    n = 10
    T = 1000
    ep = 0.7

    c_max_dir = 'c_max.csv'
    d_dir = 'd.csv'
    s_dir = 's.csv'
    desc = 'default'

    start_date = datetime(2020, 2, 16)
    end_date = datetime(2020, 4, 16)

    dates = get_dates(start_date, end_date)


    for date in dates:
        run(date, n, T, ep, c_max_dir, d_dir, s_dir, desc, save_mode='json')

if __name__ == '__main__':
    test_7()
