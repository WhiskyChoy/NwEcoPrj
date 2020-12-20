import docplex.mp.model as cpx
import math
import pandas as pd
import os
from datetime import datetime

opt_model = cpx.Model(name="MIP Model")

n = 10
range_index = range(1, n+1)

if not os.path.exists('data'):
    os.mkdir('data')

if not os.path.exists('result'):
    os.mkdir('result')

c_max_csv = open('data/c_max.csv')
d_csv = open('data/d.csv')
s_csv = open('data/s.csv')

df_c_max = pd.read_csv(c_max_csv, header=None, index_col=None).values
df_d_csv = pd.read_csv(d_csv, header=None, index_col=None).values
df_s_csv = pd.read_csv(s_csv, header=None, index_col=0)


def optimize(c_max, d, s, ep, polarize=False):
    # The decision variable is the capacity c
    c_vars = {
        (i, j):
        opt_model.integer_var(lb=0, ub=c_max[i, j], name=f'c_{i}_{j}')
        for i in range_index
        for j in range_index
        if i != j
    }

    # constraints (<= constraints).
    if polarize:
        {
            (i, j):
            opt_model.add_constraint(
                ct=s[i] * c_vars[i, j] - s[j] *
                c_vars[j, i] + math.log((1-ep)/ep) <= 0,
                ctname=f'constraint_{i}_{j}'
            )
            for i in range_index
            for j in range_index
            if i != j
        }
    else:
        {
            (i, j):
            opt_model.add_constraint(
                ct=(1 - ep) * s[i] * c_vars[i, j] -
                ep * s[j] * c_vars[j, i] <= 0,
                ctname=f'constraint_{i}_{j}'
            )
            for i in range_index
            for j in range_index
            if i != j
        }

    # objective function
    obj_func = opt_model.sum(
        c_vars[i, j]*d[i, j]
        for i in range_index
        for j in range_index
        if i != j
    )

    # optimization problem (for maximization)
    opt_model.maximize(obj_func)

    # solving with local cplex
    opt_model.solve()

    opt_df = pd.DataFrame.from_dict(
        c_vars, orient="index", columns=["variable_object"])
    opt_df.index = pd.MultiIndex.from_tuples(
        opt_df.index, names=["column_i", "column_j"])

    opt_df.reset_index(inplace=True)

    opt_df["solution_value"] = opt_df["variable_object"].apply(
        lambda item: item.solution_value)

    opt_df.drop(columns=["variable_object"], inplace=True)
    opt_df.to_csv("result/cplex_optimization_solution.csv")


def get_s_arr(date: datetime):
    str_date = f'{date.year}/{date.month}/{date.day}'
    return df_s_csv.loc[str_date].values


if __name__ == '__main__':
    # We may use the csv files in data to set up true c_max, d and s
    c_max = {(i, j): df_c_max[i-1][j-1]
             for i in range_index for j in range_index if i != j}

    d = {(i, j): df_d_csv[i-1][j-1]
         for i in range_index for j in range_index if i != j}

    s_arr = get_s_arr(datetime(2020, 4, 16))

    s = {i: s_arr[i-1] for i in range_index}

    # The epsilon here should be carefully set
    ep = 0.5
    optimize(c_max, d, s, ep)
