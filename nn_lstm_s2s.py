
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

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
        """
        # Initialize embedding
        # self.embedding = nn.Embedding(len_dict, embedding_size)

        # Bidirectional LSTM
        # self.bidirectional_LSTM = nn.LSTM(embedding_size, hidden_size = self.hidden_size, batch_first=True)

        # Sequential
        self.embedded_LSTM = nn.Sequential(
            nn.Embedding(len_dict, embedding_size),
            nn.LSTM(embedding_size, hidden_size = self.hidden_size, batch_first=True, bidirectional=True),
        )


    def forward(self, encoded_sentence : torch.Tensor):
        # Assumes that encoded_sentence is 1-d tensor, representing 1 sentence
        # Otherwise, *other x len_sentence
        # len_sentence = len(encoded_sentence) # encoded_sentence.size(-1) in general

        # Encode input
        # *other x len_sentence x embedding_size
        # word_embeddings : torch.Tensor = self.embedding(encoded_sentence)

        # LSTM
        # batch x len_sentence x 2*hidden_size, or in later pytorch, len_sentence x 2*hidden_size also 
        # This also means we must assume *other = batch (= N)
        # out_hidden_hs, _ = self.forward_LSTM(word_embeddings)
        
        # Unsqueeze added as we have no batch at the moment
        return self.embedded_LSTM(encoded_sentence.unsqueeze(0))[0]

        # This was supposed to just work...but we are using an older version of pytorch. Sigh.
        # return self.embedded_LSTM(encoded_sentence)[0]
    
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
        self.LSTM = nn.LSTM(embedding_size + self.context_size, hidden_size = self.hidden_size, batch_first=True)

        # Output
        self.out_first = DecoderDoubleOut(self.hidden_size, embedding_size, self.context_size, 2 * self.last_hidden_size)
        self.out = nn.Sequential(
            nn.MaxPool1d(2, stride=2),
            nn.Linear(self.last_hidden_size, len_dict),
            nn.LogSoftmax(-1)
        )
    
    def get_initial_states(self, encoder_outputs, device) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1 x hidden_size
        return self.init_hidden(encoder_outputs[:,0:1,self.hidden_size:]), torch.zeros(1, 1, self.hidden_size, device=device)


    def forward(self, prev_word, hidden_s, hidden_c, encoder_outputs, attn_contxts):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights
        
        Dropout (self.dropout) should be applied to the word embeddings.
        """

        # Encode (dropped-out) input
        # *other x embedding_size
        input_encoded = self.embedding(prev_word)
        # # print(f"{time.asctime()} After encoding, before attention weights!")

        # Find weighted context with attention
        # For multi-dimensional support, we would need to reshape attn_hidden
        # in order to add it properly with attn_contxt
        # 1 x hidden_size "@" (*other x ? x hidden_size + *other x len_orig_sentence x hidden_size)
        # = *other x len_orig_sentence x 1
        attn_weights = self.attn_post(self.attn_hidden(hidden_s) + attn_contxts + self.attn_bias)

        # *other x context_size x len_orig_sentence @ *other x len_orig_sentence x 1
        # = *other x context_size x 1
        context = encoder_outputs.transpose(-2, -1) @ attn_weights # encoder_outputs.mT @ attn_weights is a newer pytorch feature


        # *other x context_size
        context.squeeze_(-1)
        # # print(f"{time.asctime()} After attention weights and context, before LSTM!")

        # Concatenated Input
        # In general, *other x 1 x (embedding_size + context_size)
        concat_input = torch.cat((input_encoded, context), -1).unsqueeze(-2)
        # Concat input satisfies (N,L=1,H); in other words, assume *other = N
        # hidden_i satisfy (1,N,H)
        _, (hidden_s, hidden_c) = self.LSTM(concat_input, (hidden_s, hidden_c))
        # # print(f"{time.asctime()} After LSTM, before final layer!")
        # MaxPool requires 2d/3d, but hidden_s already satisfies this
        # *other x len_dict (because of hidden_s, assume *other = N)
        # Squeeze to avoid otherwise undesireable extra prepended (1,N,len_dict)
        word_probs = self.out(self.out_first(hidden_s.squeeze(0), input_encoded, context))

        # # print(f"{time.asctime()} After final layer, exiting decoder forward!")
        return word_probs, hidden_s, hidden_c, attn_weights

    def train_sentence(self, encoder_outputs : torch.Tensor, target_sentence : torch.Tensor, criterion):
        """
        Teaching forcing is used
        """
        # Again assume that encoder_outputs is 2d
        # otherwise shape is *other x len_orig_sentence x context_size
        # len_orig_sentence = len(encoder_outputs)
        len_target_sentence = len(target_sentence)

        # Create initial encoded start symbol, hidden states, and loss
        prev_word = torch.tensor([SOS_index], device=device)
        loss = torch.tensor(0., device=device)
        hidden_s, hidden_c = self.get_initial_states(encoder_outputs, device=device) # *other x hidden_size

        # Pre-compute partial attention weights on all encoder outputs
        # Output should be *other x len_orig_sentence x hidden_size
        attn_contxts = self.attn_contxt(encoder_outputs)
        # print(f"{time.asctime()} After setup and attn_contxt, before training loop!")

        for i in range(len_target_sentence):
            word_probs, hidden_s, hidden_c, attn_weights = self(prev_word, hidden_s, hidden_c, encoder_outputs, attn_contxts)
            # For 1D, we temporarily have to unsqueeze target_sentence to shapes to match
            loss += criterion(word_probs, target_sentence[i].unsqueeze(0))
            prev_word = target_sentence[i:i+1]
        
        
        # print(f"{time.asctime()} After training loop, exiting train_sentence!")
        return loss
    
    def decode_sentence(self, encoder_outputs : torch.Tensor):
        if self.model_select & ModelOptions.BEAM:
            return self.decode_sentence_beam(encoder_outputs)
        # Again assume that encoder_outputs is 2d
        # otherwise shape is *other x len_orig_sentence x context_size
        # len_orig_sentence = len(encoder_outputs)

        # Create initial encoded start symbol & hidden states
        prev_word = torch.tensor([SOS_index], device=device)
        hidden_s, hidden_c = self.get_initial_states(encoder_outputs, device=device) # *other x hidden_size

        # Create output list
        decoded_words : List[int] = []
        attn_weights_per_word : List[torch.Tensor] = []

        # Pre-compute partial attention weights on all encoder outputs
        # Output should be *other x len_orig_sentence x hidden_size
        attn_contxt = self.attn_contxt(encoder_outputs)
        # print(f"{time.asctime()} After setup and attn_contxt, before decoding loop!")

        for _ in self.range_max_length:
            word_probs, hidden_s, hidden_c, attn_weights = self(prev_word, hidden_s, hidden_c, encoder_outputs, attn_contxt)
            
            prev_word = word_probs.detach().topk(1)[1].squeeze(-1) #.topk(1)[1].squeeze().detach()
            # print(f"prev_word: {prev_word}")
            decoded_words.append(prev_word.squeeze())
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
        prev_word = torch.tensor([SOS_index], device=device)
        hidden_s, hidden_c = self.get_initial_states(encoder_outputs, device=device) # *other x hidden_size

        # Create output list
        candidate_sequences : List[Tuple[float, List[int], List[torch.Tensor], torch.Tensor, torch.Tensor]] = [(0., [prev_word], [], hidden_s, hidden_c)]

        final_candidate_sequences : List[Tuple[float, List[int], List[torch.Tensor], torch.Tensor, torch.Tensor]] = []

        # Pre-compute partial attention weights on all encoder outputs
        # Output should be *other x len_orig_sentence x hidden_size
        attn_contxt = self.attn_contxt(encoder_outputs)
        # print(f"{time.asctime()} After setup and attn_contxt, before decoding loop!")

        iteration = 0
        while iteration < self.max_length and len(final_candidate_sequences) <= self.search_size:
            new_candidate_sequences : List[Tuple[float, List[int], List[torch.Tensor], torch.Tensor, torch.Tensor]] = []
            for neg_log_prob, decoded_words, attn_weights_per_word, hidden_s, hidden_c in candidate_sequences:
                word_probs, hidden_s, hidden_c, attn_weights = self(torch.tensor(decoded_words[-1:], device=device), hidden_s, hidden_c, encoder_outputs, attn_contxt)

                new_attn_weights_per_word = attn_weights_per_word + [attn_weights.detach().squeeze(-1).squeeze(0)] # In place possible?

                top_log_probs, top_words = word_probs.detach().topk(self.search_size)
                
                for new_log_prob, new_word in zip(top_log_probs.squeeze(), top_words.squeeze()):
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
        
        _, decoded_words, attn_weights_per_word, _, _ = min(final_candidate_sequences) if len(final_candidate_sequences) > 1 else max(candidate_sequences)
        
        # print(f"{time.asctime()} After decoding loop, exiting decode_sentence!")
        return decoded_words[1:], torch.stack(attn_weights_per_word) # dim = -3 (*other x len_new_sentence x len_orig_sentence x 1)
