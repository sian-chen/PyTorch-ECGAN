import os
import re
import glob
import argparse

import torch
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_dir', default='./logs')
    return parser.parse_args()


def get_best_fid_from_log(log_path):
    with open(log_path) as fp:
        lines = fp.readlines()
    step, best_fid, f8, finv8, IS, last_step = -999, 999, -1, -1, -1, -1
    for line in lines[::-1]:
        if 'Best FID score' in line:
            step = int(re.search('Step: (?P<step>\d+)', line)['step'])
            best_fid = float(re.search('([0-9\.]+)$', line).group())
            break
    for line in lines:
        if str(step) in line and 'F_8 score' in line:
            f8 = float(re.search('([0-9\.]+)$', line).group())
        if str(step) in line and 'F_1/8 score' in line:
            finv8 = float(re.search('([0-9\.]+)$', line).group())
        if str(step) in line and 'Inception score' in line:
            IS = float(re.search('([0-9\.]+)$', line).group())
        if 'Step: ' in line and 'FID' in line:
            last_step = max(last_step, int(re.search('Step: (?P<step>\d+)', line)['step']))
    return step, best_fid, IS, f8, finv8, last_step


def main():
    args = parse_args()
    log_dir = args.log_dir
    cols = ['step', 'best_fid', 'IS', 'f_8', 'f_1/8', 'last_step']
    exp_names, best_fids, steps = [], [], []
    records = []
    for log_path in glob.glob(os.path.join(log_dir, '*.log')):
        exp_name = log_path.split('/')[-1]
        exp_name = exp_name[:exp_name.find('-train-')]
        record = dict(zip(cols, get_best_fid_from_log(log_path)))
        record['exp_name'] = exp_name
        records.append(record)
    df = pd.DataFrame.from_records(records, columns=['exp_name'] + cols)
    df = df.sort_values('best_fid')
    print(df)
    avg_df = df.groupby('exp_name').agg(['mean', 'std'])
    avg_df = avg_df.sort_values(('best_fid', 'mean'))
    print(avg_df)
    with pd.ExcelWriter('result.xlsx') as writer:
        df.to_excel(writer, sheet_name='full_result')
        avg_df.to_excel(writer, sheet_name='avg_result')


if __name__ == '__main__':
    main()
