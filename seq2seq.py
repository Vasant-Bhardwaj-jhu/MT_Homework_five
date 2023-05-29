#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way. 
"""


from __future__ import unicode_literals, print_function, division

from enum import Flag, auto
import argparse
import logging
import random
from re import A
import time
import itertools
from io import open
from turtle import back
from typing import Iterable, List, Optional

import matplotlib
#if you are running on the gradx/ugradx/ another cluster, 
#you will need the following line
#if you run on a local machine, you can comment it out
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu
from torch import optim

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid, 
# it can be very easy to confict with other people's jobs.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15

class ModelOptions(Flag):
    DEFAULT = 0
    TORCHLSTM = auto()
    BATCH = auto()
    BEAM = auto()
    TBPTT = auto()

    # https://stackoverflow.com/questions/43968006/support-for-enum-arguments-in-argparse
    @staticmethod
    def from_string(s):
            return ModelOptions(int(s))


from LSTM import LSTM
import hw4_s2s
import nn_lstm_s2s
import batch_s2s
import batch_nn_lstm_s2s


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index : dict[str, int] = {}
        self.word2count : dict[str, int] = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"), 
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################

def tensor_from_sentence(vocab : Vocab, sentence : str):
    """creates a tensor from a raw sentence
    """
    indexes : list[int] = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device) #.view(-1, 1)

def tensor_from_sentences(vocab : Vocab, sentences : Iterable[str]):
    """creates a tensor from raw sentences
    """
    tensors = [
        torch.tensor([vocab.word2index[word] for word in sentence.split() if word in vocab.word2index],
            dtype=torch.long, device=device)
        for sentence in sentences
    ]

    return pad_sequence(tensors, batch_first=True, padding_value=EOS_index)

    # max_len = max(len(sentence) for sentence in indices)
    # indices = [sentence + [EOS_index] * (max_len-len(sentence)) for sentence in indices]


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor

def tensors_from_pairs(src_vocab, tgt_vocab, pairs):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentences(src_vocab, (pair[0] for pair in pairs))
    target_tensor = tensor_from_sentences(tgt_vocab, (pair[1] for pair in pairs))
    return input_tensor, target_tensor


######################################################################

######################################################################

def train(input_tensor, target_tensor, encoder : torch.nn.Module , decoder : torch.nn.Module , optimizer : torch.optim.Optimizer, criterion):

    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()

    # Generate encoder output.
    # print(f"{time.asctime()} Before encoder!")
    encoder_outputs = encoder(input_tensor)
    # print(f"{time.asctime()} After encoder, before decoder!")
    
    # Run through decoder, keeping track of accumulated loss.
    optimizer.zero_grad()
    loss = decoder.train_sentence(encoder_outputs, target_tensor, criterion)
    # print(f"{time.asctime()} After decoder, before backprop!")

    # Minimize loss.
    loss.backward()
    optimizer.step()
    # print(f"{time.asctime()} After backprop, exiting train!")
    return loss.item() 



######################################################################

def translate(encoder : torch.nn.Module , decoder : torch.nn.Module , sentence, src_vocab, tgt_vocab):
    """
    runs translation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)

        # print(f"{time.asctime()} Before encoder!")
        encoder_outputs = encoder.encode_sentence(input_tensor) # Thanks Obama / If we were using 4.13...
        # print(f"{time.asctime()} After encoder, before decoder!")

        decoded_word_indices, decoder_attentions = decoder.decode_sentence(encoder_outputs)
        # print(f"{time.asctime()} After decoder, before index2word!")

        decoded_words = [tgt_vocab.index2word[index.item()] for index in decoded_word_indices]
        # print(f"{time.asctime()} After decoder, exiting translate!")

        return decoded_words, decoder_attentions


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None):
    output_sentences = []
    print(f"{time.asctime()} Before translate!")
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    print(f"{time.asctime()} After translate!")
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for _ in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

output_prefix = "TEST"
output_dir = "output"

def show_attention(input_sentence : str, output_words : List[str], attentions : torch.Tensor):
    """visualize the attention mechanism. And save it to a file. 
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    
    "*** YOUR CODE HERE ***"
    cleaned_input_sentence = clean(input_sentence)
    cleaned_input_words = cleaned_input_sentence.split()
    cleaned_output_words = clean(' '.join(output_words)).split()

    # Note: Source / Input is French, Target / Output is English (?)
    # attentions is a tensor whose first dimension (row) is output word,
    # second dimension (column) is attention given to input word for the specific (row) corresponding output word
    # plt.matshow displays tensor in expected row / column format
    # So currently this will have Output / English on rows, Input / French on columns
    plt.matshow(attentions, cmap='gray', vmin=0, vmax=1)
    plt.xticks(range(len(cleaned_input_words)), cleaned_input_words, rotation=90)
    plt.yticks(range(len(cleaned_output_words)), cleaned_output_words)
    plt.colorbar()
    
    if len(cleaned_input_sentence) > 20:
        plt.savefig(f'{output_dir}/{output_prefix}_{cleaned_input_sentence[:20]}.png')
    else:
        plt.savefig(f'{output_dir}/{output_prefix}_{cleaned_input_sentence}.png')


def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    # print('attentions =', attentions.size())
    show_attention(input_sentence, output_words, attentions)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    global output_dir
    global output_prefix

    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=10000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=500, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')
    ap.add_argument('--output_dir', default=output_dir,
                    help='output folder for attention images')
    ap.add_argument('--output_prefix', default=None,
                    help='output prefix for attention images')
    ap.add_argument('--model_select', type=ModelOptions.from_string, default=ModelOptions.TORCHLSTM | ModelOptions.BATCH,
                    help=(
                        "Select from different models: "
                        "bit 0 controls self vs torch LSTM, "
                        "bit 1 controls batching, "
                        "bit 2 controls greedy vs beam search, "
                        "bit 3 controls full vs trunctated BPTT"
                    ))
    ap.add_argument('--torch_lstm', action='store_true', default=False)
    ap.add_argument('--batching', default=64, type=int)
    ap.add_argument('--batch_order', default=0, type=int)
    ap.add_argument('--beam_search', action='store_true', default=False)
    ap.add_argument('--search_size', default=1, type=int)
    ap.add_argument('--trunc_bptt', action='store_true', default=False)

    args = ap.parse_args()

    if args.model_select:
        if (args.torch_lstm or args.beam_search):
            ap.error("--model_select is mutually exclusive with standalone option flags!")
        if 0 > args.model_select.value > 7:
            ap.error("--model_select is out of range! 0-7 (inclusive) integers are valid")
        model_select = args.model_select
    else:
        model_select = ModelOptions.DEFAULT
        if args.torch_lstm:
            model_select |= ModelOptions.TORCHLSTM
        if args.batching != 1:
            model_select |= ModelOptions.BATCH
        if args.beam_search:
            model_select |= ModelOptions.BEAM
        if args.trunc_bptt:
            ap.error("--beam_search is not yet implemented!")
            model_select |= ModelOptions.TBPTT

    # Establish directory & prefix for output images
    output_dir = args.output_dir 
    if args.output_prefix is None:
        output_prefix = time.strftime("%Y%m%d%H%M%S_", time.localtime())
    else:
        output_prefix = args.output_prefix

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration 
    if args.load_checkpoint is not None:
        print("Loading checkpoint!")
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    if model_select.value % ModelOptions.BEAM.value == ModelOptions.DEFAULT.value:
        encoder = hw4_s2s.EncoderRNN
        decoder = hw4_s2s.AttnDecoderRNN
        state_prefix = "default"
    elif model_select.value % ModelOptions.BEAM.value == ModelOptions.TORCHLSTM.value:
        encoder = nn_lstm_s2s.EncoderRNN
        decoder = nn_lstm_s2s.AttnDecoderRNN
        state_prefix = "nn_lstm"
    elif model_select.value % ModelOptions.BEAM.value == ModelOptions.BATCH.value:
        encoder = batch_s2s.EncoderRNN
        decoder = batch_s2s.AttnDecoderRNN
        state_prefix = "batch"
    elif model_select.value % ModelOptions.BEAM.value == (ModelOptions.TORCHLSTM | ModelOptions.BATCH).value:
        encoder = batch_nn_lstm_s2s.EncoderRNN
        decoder = batch_nn_lstm_s2s.AttnDecoderRNN
        state_prefix = "batch_nn_lstm"
    else:
        ap.error("Not sure how this happened, but...the selected model is not implemented")
    
    encoder = encoder(src_vocab.n_words, args.hidden_size).to(device)
    decoder = decoder(tgt_vocab.n_words, args.hidden_size, dropout_p=0.1, model_select=model_select, search_size=args.search_size).to(device)

    # encoder/decoder weights are randomly initialized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)
    
    if args.batch_order == 1:
        train_pairs.sort(key=lambda pair: len(pair[0].split()))
    elif args.batch_order == 2:
        train_pairs.sort(key=lambda pair: len(pair[1].split()))
    elif args.batch_order == 3:
        train_pairs.sort(key=lambda pair: (len(pair[0].split()), len(pair[1].split())))
    elif args.batch_order == 4:
        train_pairs.sort(key=lambda pair: (len(pair[1].split()), len(pair[0].split())))

    # set up optimization/loss
    # params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    params = itertools.chain(encoder.parameters(), decoder.parameters())
    optimizer = optim.RAdam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    upper_bound = len(train_pairs) - 2 * args.batching
    print(f"upper_bound: {upper_bound}")

    while iter_num < args.n_iters:
        iter_num += 1
        if model_select & ModelOptions.BATCH:
            if args.batch_order:
                low_slice = random.randint(0, upper_bound)
                training_pair = tensors_from_pairs(src_vocab, tgt_vocab, random.sample(train_pairs[low_slice:low_slice+2*args.batching], args.batching))
            else:
                training_pair = tensors_from_pairs(src_vocab, tgt_vocab, random.sample(train_pairs, args.batching))
        else:
            training_pair = tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs))
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = f'state_{state_prefix}_{iter_num:010d}.pt'
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            print(f"{time.asctime()} After logging begin, before translating dev set!")
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, max_num_sentences=10)
            print(f"{time.asctime()} After translating dev set, before bleu and next iteration!")

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab)


if __name__ == '__main__':
    main()
