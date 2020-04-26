from argparse import ArgumentParser, Namespace
import torch
from solver_munit import Solver_MUNIT
import yaml 
import sys
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d', default='/home/b04020/2018_autumn/IEMOCAP_full_release/data/h5py/dataset/')
    parser.add_argument('-train_set', default='train')
    parser.add_argument('-logdir', default='log/')
    
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_opt', action='store_true')

    parser.add_argument('-store_model_path', default='model_MUNIT')
    parser.add_argument('-load_model_path', default='model/')
    parser.add_argument('-summary_steps', default=100, type=int)
    parser.add_argument('-save_steps', default=1000, type=int)
    parser.add_argument('-tag', '-t', default='init')
    parser.add_argument('-iters', default=200000, type=int)

    parser.add_argument('--use_munit', action='store_true')


    args = parser.parse_args()
    
    # load config file 
    with open(args.config) as f:
        config = yaml.load(f)

    os.makedirs(args.store_model_path,exist_ok=True)
    save_sample_path = os.path.join(args.store_model_path, 'sample')
    os.makedirs(save_sample_path,exist_ok=True)
    
    if args.use_munit:
        solver = Solver_MUNIT(config=config, args=args)

    if args.iters > 0:
        solver.train(n_iterations=args.iters)
