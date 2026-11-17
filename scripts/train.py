#!/usr/bin/env python3
import argparse
from reliability_st.util import load_config,device_from_arg
from reliability_st.engine import run_cross_validation

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--device',default='auto')
    args=parser.parse_args()
    print(run_cross_validation(load_config(args.config),device_from_arg(args.device)))
