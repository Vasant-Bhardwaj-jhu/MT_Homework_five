
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from LSTM import LSTM

# Unsafe transfering of global variable(s) from calling module (seq2seq.py)
# We do it anyway to avoid adding device as a class field, to keep the same
# programming style from when all the classes were defined in one file
from __main__ import device, MAX_LENGTH, SOS_index, EOS_index, ModelOptions

class DecoderDoubleOut(nn.Module):
    def __init__(self, hidden_size : int, embedding_size : int, context_size : int, double_last_hidden_size : int):
        super().__init__()

        self.hidden_out    = nn.Linear(hidden_size,    double_last_hidden_size, bias=False)
        self.embedding_out = nn.Linear(embedding_size, double_last_hidden_size, bias=False)
        self.context_out   = nn.Linear(context_size,   double_last_hidden_size, bias=False)

        self.bias = nn.Parameter(torch.randn(double_last_hidden_size))

    def forward(self, hidden_s : torch.Tensor, input_encoded : torch.Tensor, context : torch.Tensor):
        return (
            self.hidden_out(hidden_s)
          + self.embedding_out(input_encoded)
          + self.context_out(context)
          + self.bias
        )


class EncoderRNN(nn.Module):
    """the class for the encoder RNN
    """
    def __init__(self, len_dict : int, embedding_size : int, hidden_size : Optional[int] = None):
        # super(EncoderRNN, self).__init__() # Pre python-3.3 style
        super().__init__()
        self.hidden_size = hidden_size or embedding_size # Abuse of falsy-nature of None, 0 is invalid anyway

        """Initialize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM. 
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """
        # Initialize embedding
        self.embedding = nn.Embedding(len_dict, embedding_size)

        # Forward LSTM
        self.forward_LSTM = LSTM(embedding_size, hidden_size = self.hidden_size)
        # Backward LSTM
        self.backward_LSTM = LSTM(embedding_size, hidden_size = self.hidden_size)


    def forward(self, encoded_sentence : torch.Tensor):
        # Assumes that encoded_sentence is 1-d tensor, representing 1 sentence
        # Otherwise, *other x len_sentence
        size_sentence = encoded_sentence.size()
        len_sentence = size_sentence[-1] # Assumes that encoded_sentence was not reshaped to "column vectors"

        # Encode input
        # *other x len_sentence x embedding_size
        word_embeddings : torch.Tensor = self.embedding(encoded_sentence)
        # assert list(word_embeddings.size()) == list(encoded_sentence.size()) + [self.embedding_size]

        # Create output tensor
        # *other x len_sentence x 2*hidden_size
        out_hidden_hs = torch.zeros(*size_sentence[0:-1], len_sentence, 2 * self.hidden_size, device=device)
        forward_out  = out_hidden_hs.narrow(-1, 0,                self.hidden_size)
        backward_out = out_hidden_hs.narrow(-1, self.hidden_size, self.hidden_size)
        # print(f"{time.asctime()} After embedding, and setup, before forward!")

        # LSTM, forward

        # For multi-dimensional support, we can try the following:
        # parallel_word_embeddings = (word_embeddings.select(-2, i) for i in range(word_embeddings.size(-2)))
        # for word in parallel_word_embeddings:
        #     pass # etc.
        # But given that this only easily works with sentences with exactly the same
        # length, this is not attempted
        # ...until now

        f_hidden_h, f_hidden_c = self.forward_LSTM.get_initial_states(device=device)
        for i in range(len_sentence):
            f_hidden_h, f_hidden_c = self.forward_LSTM(f_hidden_h, f_hidden_c, word_embeddings.select(-2, i))
            this_step_forward_out = forward_out.select(-2, i)
            this_step_forward_out[:] = f_hidden_h

        # print(f"{time.asctime()} After forward, before backwards!")

        # LSTM, backward
        b_hidden_h, b_hidden_c = self.backward_LSTM.get_initial_states(device=device)
        for i in reversed(range(len_sentence)):
            b_hidden_h, b_hidden_c = self.backward_LSTM(b_hidden_h, b_hidden_c, word_embeddings.select(-2, i))
            this_step_backward_out = backward_out.select(-2, i)
            this_step_backward_out[:] = b_hidden_h


        # print(f"{time.asctime()} After backwards, exiting encoder!")
        return out_hidden_hs
    
    def encode_sentence(self, encoded_sentence : torch.Tensor):
        return self(encoded_sentence)


class AttnDecoderRNN(nn.Module):
    """the class for the decoder 
    """
    def __init__(self, len_dict : int,  embedding_size : int,
                                        context_size : Optional[int] = None,
                                        hidden_size : Optional[int] = None,
                                        last_hidden_size : Optional[int] = None,
                                        dropout_p : float = 0.1,
                                        model_select : ModelOptions = ModelOptions.DEFAULT,
                                        search_size : int = 3,
                                        max_length : int = MAX_LENGTH):
        # super(AttnDecoderRNN, self).__init__() # Pre python-3.3 style
        super().__init__()
        self.hidden_size = hidden_size or embedding_size # Abuse of falsy-nature of None, 0 is invalid anyway
        self.context_size = context_size or 2 * self.hidden_size
        self.last_hidden_size = last_hidden_size or (self.hidden_size + 1) // 2
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.range_max_length = range(max_length)
        self.model_select = model_select
        self.search_size = search_size
        
        """Initialize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        
        # Embedding
        self.embedding = nn.Sequential(
            nn.Embedding(len_dict, embedding_size),
            nn.Dropout(self.dropout_p),
        )

        # Init hidden state
        self.init_hidden = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        ) 

        # Attention
        self.attn_hidden = nn.Linear(self.hidden_size,  self.hidden_size, bias=False)
        self.attn_contxt = nn.Linear(self.context_size, self.hidden_size, bias=False)
        self.attn_bias = nn.Parameter(torch.randn(self.hidden_size))
        self.attn_post = nn.Sequential(
            # nn.Linear(self.context_size + self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
            nn.Softmax(-2), # nn.Linear, even if output_size = 1, will create a vacuous dimension
        )

        # LSTM
        self.LSTM = LSTM(embedding_size, self.context_size, hidden_size = self.hidden_size)

        # Output
        self.out_first = DecoderDoubleOut(self.hidden_size, embedding_size, self.context_size, 2 * self.last_hidden_size)
        self.out = nn.Sequential(
            nn.MaxPool1d(2, stride=2),
            nn.Linear(self.last_hidden_size, len_dict),
            nn.LogSoftmax(-1)
        ) 


    def forward(self, prev_word, hidden_s, hidden_c, encoder_outputs, attn_contxts):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights
        
        Dropout (self.dropout) should be applied to the word embeddings.
        """

        # Encode (dropped-out) input
        # In 2d, embedding_size, but in general, *other x embedding_size
        input_encoded = self.embedding(prev_word)
        # # print(f"{time.asctime()} After encoding, before attention weights!")

        # Find weighted context with attention
        # attn_hidden = self.attn_hidden(hidden_s) # *other x hidden_size

        # For multi-dimensional support, we would need to reshape attn_hidden
        # in order to add it properly with attn_contxt
        # 1 x hidden_size "@" (*other x ? x hidden_size + *other x len_orig_sentence x hidden_size)
        # = *other x len_orig_sentence x 1
        attn_weights = self.attn_post(self.attn_hidden(hidden_s).unsqueeze(-2) + attn_contxts + self.attn_bias)

        # *other x context_size x len_orig_sentence @ *other x len_orig_sentence x 1
        # = *other x context_size x 1
        context = encoder_outputs.transpose(-2, -1) @ attn_weights # encoder_outputs.mT @ attn_weights is a newer pytorch feature


        # *other x context_size
        context.squeeze_(-1)
        # # print(f"{time.asctime()} After attention weights and context, before LSTM!")

        # *other x hidden_size
        hidden_s, hidden_c = self.LSTM(hidden_s, hidden_c, input_encoded, context)
        # # print(f"{time.asctime()} After LSTM, before final layer!")
        # MaxPool requires 2d/3d, but context already satisfies this
        # nn.Sequential cannot handle multiple positional arguments, so...
        # *other x len_dict
        word_probs = self.out(self.out_first(hidden_s, input_encoded, context))

        # # print(f"{time.asctime()} After final layer, exiting decoder forward!")
        return word_probs, hidden_s, hidden_c, attn_weights

    def train_sentence(self, encoder_outputs : torch.Tensor, target_sentence : torch.Tensor, criterion):
        """
        Teaching forcing is used
        Awkward that target_sentence is really now target_sentences...
        """
        # Again assume that encoder_outputs is 2d
        # otherwise shape is *other x len_orig_sentence x context_size
        # len_orig_sentence = len(encoder_outputs)
        len_target_sentence = target_sentence.size(-1)

        # Create initial encoded start symbol, hidden states, and loss
        prev_word = torch.tensor(SOS_index, device=device)
        loss = torch.tensor(0., device=device)
        hidden_s = self.init_hidden(encoder_outputs.select(-2, 0).narrow(-1, self.hidden_size, self.hidden_size)) # *other x hidden_size
        _, hidden_c = self.LSTM.get_initial_states(device=device)

        # Pre-compute partial attention weights on all encoder outputs
        # Output should be *other x len_orig_sentence x hidden_size
        attn_contxts = self.attn_contxt(encoder_outputs)
        # print(f"{time.asctime()} After setup and attn_contxt, before training loop!")

        for i in range(len_target_sentence):
            word_probs, hidden_s, hidden_c, attn_weights = self(prev_word, hidden_s, hidden_c, encoder_outputs, attn_contxts)
            loss += criterion(word_probs, target_sentence.select(-1, i))
            prev_word = target_sentence.select(-1, i)
        
        
        # print(f"{time.asctime()} After training loop, exiting train_sentence!")
        return loss

    def train_sentence_trunc(self, encoder_outputs : torch.Tensor, target_sentence : torch.Tensor, criterion):
        """
        Teaching forcing is used
        """
        # Again assume that encoder_outputs is 2d
        # otherwise shape is *other x len_orig_sentence x context_size
        # len_orig_sentence = len(encoder_outputs)
        len_target_sentence = target_sentence.size(-1)

        # Create initial encoded start symbol, hidden states, and loss
        prev_word = torch.tensor(SOS_index, device=device)
        loss = torch.tensor(0., device=device)
        hidden_s = self.init_hidden(encoder_outputs.select(-2, 0).narrow(-1, self.hidden_size, self.hidden_size)) # *other x hidden_size
        _, hidden_c = self.LSTM.get_initial_states(device=device)

        # Pre-compute partial attention weights on all encoder outputs
        # Output should be *other x len_orig_sentence x hidden_size
        attn_contxts = self.attn_contxt(encoder_outputs)
        # print(f"{time.asctime()} After setup and attn_contxt, before training loop!")

        for i in range(len_target_sentence):
            word_probs, hidden_s, hidden_c, attn_weights = self(prev_word, hidden_s, hidden_c, encoder_outputs, attn_contxts)
            loss += criterion(word_probs, target_sentence.select(-1, i))
            prev_word = target_sentence.select(-1, i)
        
        
        # print(f"{time.asctime()} After training loop, exiting train_sentence!")
        return loss.detatch()
    
    def decode_sentence(self, encoder_outputs : torch.Tensor):
        if self.model_select & ModelOptions.BEAM:
            return self.decode_sentence_beam(encoder_outputs)

        # Again assume that encoder_outputs is 2d
        # otherwise shape is *other x len_orig_sentence x context_size
        # len_orig_sentence = len(encoder_outputs)

        # Create initial encoded start symbol & hidden states
        prev_word = torch.tensor(SOS_index, device=device)
        hidden_s = self.init_hidden(encoder_outputs[0,self.hidden_size:]) # *other x hidden_size
        _, hidden_c = self.LSTM.get_initial_states(device=device)

        # Create output list
        decoded_words : List[int] = []
        attn_weights_per_word : List[torch.Tensor] = []

        # Pre-compute partial attention weights on all encoder outputs
        # Output should be *other x len_orig_sentence x hidden_size
        attn_contxt = self.attn_contxt(encoder_outputs)
        # print(f"{time.asctime()} After setup and attn_contxt, before decoding loop!")

        hidden_s.unsqueeze_(0), hidden_c.unsqueeze_(0), encoder_outputs.unsqueeze_(0)

        for _ in self.range_max_length:
            word_probs, hidden_s, hidden_c, attn_weights = self(prev_word, hidden_s, hidden_c, encoder_outputs, attn_contxt)
            
            prev_word = word_probs.detach().topk(1)[1].squeeze() #.topk(1)[1].squeeze().detach()
            # print(f"prev_word: {prev_word}")
            decoded_words.append(prev_word)
            attn_weights_per_word.append(attn_weights.detach().squeeze(-1).squeeze(0)) # Batch = 1
            if prev_word.item() == EOS_index:
                break
        
        # print(f"{time.asctime()} After decoding loop, exiting decode_sentence!")
        return decoded_words, torch.stack(attn_weights_per_word) # dim = -3 (*other x len_new_sentence x len_orig_sentence x 1)
    
    def decode_sentence_beam(self, encoder_outputs : torch.Tensor):
        # Again assume that encoder_outputs is 2d
        # otherwise shape is *other x len_orig_sentence x context_size
        # len_orig_sentence = len(encoder_outputs)

        # Create initial encoded start symbol & hidden states
        prev_word = torch.tensor(SOS_index, device=device)
        hidden_s = self.init_hidden(encoder_outputs[0,self.hidden_size:]) # *other x hidden_size
        _, hidden_c = self.LSTM.get_initial_states(device=device)

        # # Create output list

        # For beam search, we have to keep track of attention that generated
        # & each word in the sequence, the probability of the sequence, and
        # The hidden states to use to continue the sequence. That's 4 things,
        # prob being a scalar, hidden state being a 1d tensor through hidden_size,
        # words being a 1d tensor through time, and attention being a 2d tensor
        # through time and hidden_size
        # neg_log_prob_sequences : List[float] = [0.]
        # decoded_words_sequences : List[List[int]] = [[prev_word]]
        # attn_weights_per_word_sequences : List[List[torch.Tensor]] = [[]]
        # hidden_state_sequences : List[Tuple[torch.Tensor, torch.Tensor]] = [(hidden_s, hidden_c)]
        candidate_sequences : List[Tuple[float, List[int], List[torch.Tensor], torch.Tensor, torch.Tensor]] = [(0., [prev_word], [], hidden_s, hidden_c)]

        final_candidate_sequences : List[Tuple[float, List[int], List[torch.Tensor], torch.Tensor, torch.Tensor]] = []

        # Pre-compute partial attention weights on all encoder outputs
        # Output should be *other x len_orig_sentence x hidden_size
        attn_contxt = self.attn_contxt(encoder_outputs)
        # print(f"{time.asctime()} After setup and attn_contxt, before decoding loop!")

        hidden_s.unsqueeze_(0), hidden_c.unsqueeze_(0), encoder_outputs.unsqueeze_(0)

        iteration = 0
        # start = time.time()
        while iteration < self.max_length and len(final_candidate_sequences) <= self.search_size:
            new_candidate_sequences : List[Tuple[float, List[int], List[torch.Tensor], torch.Tensor, torch.Tensor]] = []
            for neg_log_prob, decoded_words, attn_weights_per_word, hidden_s, hidden_c in candidate_sequences:
                # prev_word = decoded_words[-1]

                word_probs, hidden_s, hidden_c, attn_weights = self(decoded_words[-1], hidden_s, hidden_c, encoder_outputs, attn_contxt)

                new_attn_weights_per_word = attn_weights_per_word + [attn_weights.detach().squeeze(-1).squeeze(0)] # In place possible?
                
                top_log_probs, top_words = word_probs.detach().topk(self.search_size)
                
                for new_log_prob, new_word in zip(top_log_probs.squeeze(), top_words.squeeze()):
                    # print(f"prev_word: {prev_word}")
                    new_seq_tuple = (
                        neg_log_prob - new_log_prob,
                        decoded_words + [new_word],
                        new_attn_weights_per_word,
                        hidden_s,
                        hidden_c
                    )
                    if new_word.item() == EOS_index:
                        final_candidate_sequences.append(new_seq_tuple)
                    else:
                        new_candidate_sequences.append(new_seq_tuple)
            
            candidate_sequences = sorted(new_candidate_sequences)[:self.search_size]
            iteration += 1
            # end = time.time()
            # print(f"{iteration}, {end-start}")
            # start = time.time()
        
        # print(final_candidate_sequences)
        _, decoded_words, attn_weights_per_word, _, _ = min(final_candidate_sequences) if len(final_candidate_sequences) > 1 else max(candidate_sequences)

        # print(f"{time.asctime()} After decoding loop, exiting decode_sentence!")
        return decoded_words[1:], torch.stack(attn_weights_per_word) # dim = -3 (*other x len_new_sentence x len_orig_sentence x 1)


    def decode_sentences(self, encoder_outputs : torch.Tensor):
        # Again assume that encoder_outputs is 2d
        # otherwise shape is *other x len_orig_sentence x context_size
        # len_orig_sentence = len(encoder_outputs)

        # Create initial encoded start symbol & hidden states
        prev_word = torch.tensor(SOS_index, device=device)
        hidden_s = self.init_hidden(encoder_outputs.select(-2, 0).narrow(-1, self.hidden_size, self.hidden_size)) # *other x hidden_size
        _, hidden_c = self.LSTM.get_initial_states(device=device)

        # Create output list
        decoded_words : List[int] = []
        attn_weights_per_word : List[torch.Tensor] = []

        # Pre-compute partial attention weights on all encoder outputs
        # Output should be *other x len_orig_sentence x hidden_size
        attn_contxt = self.attn_contxt(encoder_outputs)
        # print(f"{time.asctime()} After setup and attn_contxt, before decoding loop!")

        for _ in self.range_max_length:
            word_probs, hidden_s, hidden_c, attn_weights = self(prev_word, hidden_s, hidden_c, encoder_outputs, attn_contxt)
            
            prev_word = word_probs.detach().topk(1)[1].squeeze() #.topk(1)[1].squeeze().detach()
            # print(f"prev_word: {prev_word}")
            decoded_words.append(prev_word)
            attn_weights_per_word.append(attn_weights.detach().squeeze(-1))
            if prev_word.item() == EOS_index:
                break
        
        # print(f"{time.asctime()} After decoding loop, exiting decode_sentence!")
        return decoded_words, torch.stack(attn_weights_per_word) # dim = -3 (*other x len_new_sentence x len_orig_sentence x 1)
