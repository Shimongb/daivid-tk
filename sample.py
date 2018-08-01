# coding=utf-8
# encoding=utf8

from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model


def main(prime_str='', num_of_words=52, i_encode="utf-8", print_res=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to load stored checkpointed models from')
    parser.add_argument('-n', type=int, default=num_of_words,
                        help='number of words to sample')
    parser.add_argument('--prime', type=str, default=prime_str,
                        help='prime text')
    parser.add_argument('--pick', type=int, default=1,
                        help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', type=int, default=4,
                        help='width of the beam search')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    parser.add_argument('--count', '-c', type=int, default=1,
                        help='number of samples to print')
    parser.add_argument('--quiet', '-q', default=True, action='store_true',
                        help='suppress printing the prime text (default false)')

    args = parser.parse_args()
    return sample(args, i_encode, print_res)


def sample(args, i_encode, print_res=True):
    res_text = ''
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f, encoding=i_encode)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for _ in range(args.count):
                res_text += model.sample(sess, words, vocab, args.n, args.prime, args.sample, args.pick, args.width,
                                         args.quiet)
                # print(model.sample(sess, words, vocab, args.n, args.prime, args.sample, args.pick, args.width, args.quiet))
    if print_res:
        print(' *** ')
        print(res_text)
        print(' *** ')
    return res_text


if __name__ == '__main__':
    main()
