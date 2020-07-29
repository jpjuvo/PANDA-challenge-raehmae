import sys
sys.path.append('.')
from evaluation.eval_funcs import evaluate_model_dir
from model.model_config import ModelConfig 
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description="Evaluate model folder")
parser.add_argument('--model_dir', help="directory containing the models", required=True)
parser.add_argument('--train_image_dir', help="images folder", required=True)
parser.add_argument('--train_csv', help="path to train csv", required=True)
parser.add_argument('--mean', nargs='+', help='list of mean values, e.g. --mean 0.0905 0.1811 0.1220', type=float, required=True)
parser.add_argument('--std', nargs='+', help='list of std values, e.g. --std 0.3635, 0.4998, 0.4047', type=float, required=True)
parser.add_argument('--model_name', help="model name identifier, e.g. iafoss", default="iafoss" ,required=False)
parser.add_argument('--arch', help="encoder architecture, e.g. resnext50_32x4d_ssl", default="resnext50_32x4d_ssl" ,required=False)
parser.add_argument('--model_n_out', help="output dimension of the model", default=6, type=int, required=False)
parser.add_argument('--sz', help="tile size", default=128, type=int, required=False)
parser.add_argument('--N', help="mosaic tile count", type=int, default=12)
parser.add_argument('--model_file_prefix', help="name part of the model before index, e.g. RNXT50_", default="")



FLAGS = parser.parse_args()

def main():
    mean = FLAGS.mean
    std = FLAGS.std

    config_path = os.path.join(FLAGS.model_dir,"config.json")
    if not os.path.isfile(config_path):
        # create the model config
        config = ModelConfig(model_name=FLAGS.model_name,
                            arch=FLAGS.arch, 
                            model_n_out=FLAGS.model_n_out,
                            sz=FLAGS.sz,
                            N=FLAGS.N,
                            mean=np.array(mean),
                            std=np.array(std),
                            meta={
                                "model_file_prefix":FLAGS.model_file_prefix
                            })
        config.toDir(FLAGS.model_dir)

    evaluate_model_dir(FLAGS.model_dir, sampler=None, TRAIN=FLAGS.train_image_dir, LABELS=FLAGS.train_csv)

main()