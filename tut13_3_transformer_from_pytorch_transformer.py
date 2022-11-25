### Program implementing transformer- version 4 (uses pytorch's internal nn.Transformer module)

## Features:
# 1. Two necessary requirements for this implementation to work: (a) pad_token = 0 (automatically masks values) (b) keys = values (typically true for attention by definition)
# 2. Modifications over version 2 (tut13_1_transformer_from_scratch_v2.py):
# 2.1. Uses pytroch's inbuilt nn.Transformer
# 2.2. uses torch.repeat instead of torch.expand to broadcast masks (torch.repeat is more correct when broadcasting to dimensions of size > 1, according to pytorch forum)

## Todos / Questions:
# 1.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

# utility function to create N copies of a module as a list (note: not sequential)
def clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

# utility function to create upper triangular mask for decoder masked attention
def subsequent_mask(mask_shape, n_heads):
    batch_size, max_seq_len = mask_shape
    mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).type(torch.uint8) # mask.shape = [max_seq_len, max_seq_len]
    # expand dims of mask to match the mask_shape expected by nn.Transformer: [batch_size * n_heads, seq_len, seq_len]
    mask = mask.repeat(batch_size * n_heads, 1, 1) # mask.shape = [batch_size * n_heads, max_seq_len, max_seq_len]
    return mask == 1 # True elements are masked

# utility function to create mask over pad tokens
def pad_mask(keys, pad_token, n_heads):
    batch_size, max_seq_len = keys.shape
    mask = keys.eq(pad_token) # mask.shape: [batch_size, max_seq_len]
    # expand dims of mask to match the mask_shape expected by nn.Transformer: [batch_size * n_heads, seq_len, seq_len]
    mask = mask.unsqueeze(1) # mask.shape: [batch_size, 1, max_seq_len]
    mask = mask.repeat(1, max_seq_len, 1) # mask.shape: [batch_size, max_seq_len, max_seq_len]
    mask = mask.repeat(n_heads, 1, 1) # mask.shape = [batch_size * n_heads, max_seq_len, max_seq_len]
    return mask # True elements are masked

# utility function to calculate train loss (cross entropy loss)
def calculate_loss(model, minibatch, pad_token, n_heads, device):
    enc_input, dec_input, target = minibatch[:,0], minibatch[:,1], minibatch[:,2] # minibatch.shape: [batch_size, 3, max_seq_len]
    enc_input, dec_input, target = torch.LongTensor(enc_input).to(device), torch.LongTensor(dec_input).to(device), torch.tensor(target).to(device) # convert to tensors

    # pad masking
    # pad masking is applied only on the basis of keys (no need to refer queries - this works out because keys = values in attention)
    enc_enc_mask = pad_mask(enc_input, pad_token, n_heads)
    dec_dec_mask_padding = pad_mask(dec_input, pad_token, n_heads)
    enc_dec_mask = enc_enc_mask # since keys = enc_input for cross attention

    # create subsequent masking for dec_input
    dec_dec_mask_causal = subsequent_mask(dec_input.shape, n_heads).to(device)
    # combine dec_dec_mask_padding and dec_dec_mask_causal (using logical_or, since True values are masked)
    dec_dec_mask = torch.logical_or(dec_dec_mask_padding, dec_dec_mask_causal)

    # forward prop through model - note that we do not pass enc_dec_mask as its equal to enc_enc_mask
    scores = model(enc_input, enc_enc_mask, dec_input, dec_dec_mask) # scores.shape: [batch_size, max_seq_len, tgt_vocab]
    scores = scores.transpose(1, 2) # scores.shape: [batch_size, tgt_vocab, max_seq_len]
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss = criterion(scores, target)
    return loss

# class to obtain embeddings from vocab tokens; embed_dim = d_model
class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model) # the embedding vector is multiplied by sqrt(d_model)

# class to add positional encoding to embeddings (note that this class implements positional encoding as a constant untrainable vector)
class PositionalEncoding_Fixed(nn.Module):
    def __init__(self, d_model, dropout, maxlen=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # calculate positional encoding and save them (register buffer for saving params in params_dict that are not to be updated during backprop)
        pe = torch.zeros(maxlen, d_model)
        pos = torch.arange(maxlen).unsqueeze(1) # pos.shape: [maxlen, 1]
        div_term = 10000.0 * torch.exp( torch.arange(0, d_model, 2) / d_model ) # div_term.shape: [d_model/2]
        pe[:, 0::2] = torch.sin(pos / div_term) # pe[:, 0::2].shape: [maxlen, d_model/2]
        pe[:, 1::2] = torch.cos(pos / div_term)
        self.register_buffer("pe", pe)
    # add positional encoding to the embedding - and freeze positional encoding
    def forward(self, x): # x.shape: [batch_size, seq_len, d_model]
        x = x + self.pe[x.size(1), :].requires_grad_(False) # but there are no trainable weights in positional encoding anyway ?!
        return self.dropout(x) # whats the use / effect of dropout in positional embeddings ?

# class to add positional encoding to embeddings (note that this class implements learnable positional encoding)
class PositionalEncoding_Learnt(nn.Module):
    def __init__(self, d_model, dropout, device):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.device = device
    # add positional encoding to the embedding
    def forward(self, x): # x.shape: [batch_size, seq_len, d_model]
        batch_size, max_seq_len = x.shape[0], x.shape[1]
        positions = torch.arange(max_seq_len).to(self.device)
        positions = positions.unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        pos_emb = self.pos_emb(positions)
        final_emb = self.dropout( self.norm(x + pos_emb) )
        return final_emb # whats the use / effect of dropout in positional embeddings ?


# class implemeting the entire transformer (using nn.Transformer)
class EncoderDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout, f_src_emb, f_tgt_emb, tgt_vocab_size, device):
        super().__init__()
        self.transformer = nn.Transformer(d_model=d_model,
                                        nhead=n_heads,
                                        num_encoder_layers=n_layers,
                                        num_decoder_layers=n_layers,
                                        dim_feedforward=d_ff,
                                        dropout=dropout,
                                        activation='gelu',
                                        batch_first=True,
                                        norm_first=True, # True -> layernorm at input; False -> layernorm at output
                                        device=device)
        self.f_src_emb = f_src_emb
        self.f_tgt_emb = f_tgt_emb
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, src, src_mask, tgt, tgt_mask):
        # convert tokens to embeddings
        src = self.f_src_emb(src)
        tgt = self.f_tgt_emb(tgt)
        # forward prop through transformer
        decoder_out = self.transformer(src,
                                        tgt,
                                        src_mask=src_mask, # src_mask.shape: [batch_size * n_heads, seq_len, seq_len]
                                        tgt_mask=tgt_mask) # tgt_mask.shape: [batch_size * n_heads, seq_len, seq_len]

        scores = self.projection(decoder_out) # scores.shape: [batch_size, seq_len, tgt_vocab]
        return scores

    # function used for test time generation - outputs score only for last element in seq
    def generate(self, src, src_mask, tgt, tgt_mask, pred_index): # decoder_out.shape: [batch_size, seq_len, d_model]
        src = self.f_src_emb(src)
        tgt = self.f_tgt_emb(tgt)
        decoder_out = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        score = self.projection(decoder_out[:, pred_index]) # score.shape: [batch_size, tgt_vocab]
        return decoder_out, score


# caller function to instantiate the transformer model, using the defined hyperparams as input
def make_model(src_vocab_size, tgt_vocab_size, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device):
    # posenc = PositionalEncoding_Fixed(d_model, dropout) # positional encoding block (fixed)
    posenc = PositionalEncoding_Learnt(d_model, dropout, device) # positional encoding block (learnt)
    f_src_emb = nn.Sequential(Embeddings(src_vocab_size, d_model), deepcopy(posenc)) # function to convert src tokens into src embeddings
    f_tgt_emb = nn.Sequential(Embeddings(tgt_vocab_size, d_model), deepcopy(posenc)) # function to convert tgt tokens into tgt embeddings
    model = EncoderDecoder(d_model, n_heads, n_layers, d_ff, dropout, f_src_emb, f_tgt_emb, tgt_vocab_size, device) # the transformer model: encoder-decoder pair combining all modules

    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


# utility function implementing greedy decoding
def greedy_decoding(encoder_input, enc_enc_mask, dec_input, max_seq_len, pad_token, n_heads):
    for i in range(max_seq_len - 1):
        dec_dec_mask_padding = pad_mask(dec_input, pad_token, n_heads)
        dec_dec_mask_causal = subsequent_mask(dec_input.shape, n_heads).to(device)
        dec_dec_mask = torch.logical_or(dec_dec_mask_padding, dec_dec_mask_causal)
        decoder_out, score = model.generate(encoder_input, enc_enc_mask, dec_input, dec_dec_mask, i) # score.shape: [1, tgt_vocab_size]
        logprobs = F.log_softmax(score, dim=-1)
        _, next_word_token = torch.max(logprobs, dim=-1)
        if torch.cuda.is_available():
            next_word_token = torch.cuda.LongTensor(next_word_token)
        else:
            next_word_token = torch.LongTensor(next_word_token)
        # update dec_input
        dec_input[0][i+1] = next_word_token
    return dec_input


# utility function implementing recursive beam search
def recursive_beam_search(curr_beam, encoder_input, enc_enc_mask, recursion_depth, max_seq_len, beam_width, pad_token, n_heads):
    all_beam_candidates = []
    # termination case
    if recursion_depth == max_seq_len:
        return curr_beam

    for dec_input, logprob in curr_beam:
        dec_dec_mask_padding = pad_mask(dec_input, pad_token, n_heads)
        dec_dec_mask_causal = subsequent_mask(dec_input.shape, n_heads).to(device)
        dec_dec_mask = torch.logical_or(dec_dec_mask_padding, dec_dec_mask_causal)
        decoder_out, score = model.generate(encoder_input, enc_enc_mask, dec_input, dec_dec_mask, recursion_depth-1) # score.shape: [1, tgt_vocab_size]
        new_logprobs = F.log_softmax(score, dim=-1).squeeze(0)

        # beam candidates: list of tuples [ updated_dec_input, updated_score ]
        for class_token, new_logprob in enumerate(new_logprobs):
            if torch.cuda.is_available():
                next_word_token = torch.cuda.LongTensor([class_token])
            else:
                next_word_token = torch.LongTensor([class_token])
            updated_dec_input = deepcopy(dec_input)
            updated_dec_input[0][recursion_depth] = next_word_token
            updated_logprob = ( (logprob * (recursion_depth-1)) + new_logprob ) / recursion_depth # length normalized logprob
            new_beam_candidate = [ updated_dec_input, updated_logprob ]
            all_beam_candidates.append(new_beam_candidate)

    # select topk candidates
    all_beam_candidates = list(sorted(all_beam_candidates, key=lambda x: x[1], reverse=True))
    new_beam = all_beam_candidates[:beam_width]
    # recursive
    final_beam = recursive_beam_search(new_beam, encoder_input, enc_enc_mask, recursion_depth+1, max_seq_len, beam_width, pad_token, n_heads)
    return final_beam


# utility function implementing beam search decoding
def beam_search_decoding(encoder_input, enc_enc_mask, dec_input, max_seq_len, beam_width, pad_token, n_heads):
    all_beam_candidates = []
    recursion_depth = 1

    dec_dec_mask_padding = pad_mask(dec_input, pad_token, n_heads)
    dec_dec_mask_causal = subsequent_mask(dec_input.shape, n_heads).to(device)
    dec_dec_mask = torch.logical_or(dec_dec_mask_padding, dec_dec_mask_causal)
    decoder_out, score = model.generate(encoder_input, enc_enc_mask, dec_input, dec_dec_mask, recursion_depth-1)  # score.shape: [1, tgt_vocab_size]
    logprobs = F.log_softmax(score, dim=-1).squeeze(0)

    # beam candidates: list of tuples [ updated_dec_input, updated_score ]
    for class_token, logprob in enumerate(logprobs):
        if torch.cuda.is_available():
            next_word_token = torch.cuda.LongTensor([class_token])
        else:
            next_word_token = torch.LongTensor([class_token])
        updated_dec_input = deepcopy(dec_input)
        updated_dec_input[0][recursion_depth] = next_word_token
        updated_logprob = logprob
        new_beam_candidate = [ updated_dec_input, updated_logprob ]
        all_beam_candidates.append(new_beam_candidate)

    # select topk candidates
    all_beam_candidates = list(sorted(all_beam_candidates, key=lambda x: x[1], reverse=True))
    initial_beam = all_beam_candidates[:beam_width]
    # recursive beam search
    final_beam = recursive_beam_search(initial_beam, encoder_input, enc_enc_mask, recursion_depth+1, max_seq_len, beam_width, pad_token, n_heads)
    # pick out the best class_token_list
    best_dec_input, best_logprob = final_beam[0]
    return best_dec_input




# main
if __name__ == '__main__':
    # hyperparams
    d_model = 512
    d_k = 64 # not used in nn.Transformer as its internally calculated: d_k = d_model // n_heads
    d_v = 64 # not used in nn.Transformer as its internally calculated: d_v = d_model // n_heads
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_seq_len = 15 # useful for padding and recursion depth in beam search based translation
    beam_width = 3 # used for beam search decoding
    batch_size = 8
    lr = 1e-4
    num_epochs = 1000
    random_seed = 10

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # given parallel corpus for MT (English to French)
    corpus = [["He is jumping on the beach .", "Il saute sur la plage ."],
              ["She is running on the road .", "Elle court sur la route ."],
              ["She is climbing a mountain .", "Elle escalade une montagne ."],
              ["He is driving a car .", "Il conduit une voiture ."],
              ["She is standing on the bed .", "Elle est debout sur le lit ."],
              ["She is jumping on the car .", "Elle saute sur la voiture ."],
              ["He is running on the mountain .", "Il court sur la montagne ."],
              ["He is climbing on the bed .", "Il grimpe sur le lit ."],
              ["She is driving on the road .", "Elle roule sur la route ."],
              ["He is standing on the beach .", "Il est debout sur la plage ."]]

    ### create vocabularies and dataset

    dataset = []
    src_vocab, tgt_vocab, tgt_idx_vocab = {}, {}, {}
    n_unique_src_words, n_unique_tgt_words = 0, 0

    # add pad token in both src_vocab and tgt_vocab
    pad_token = 0
    src_vocab['<P>'] = pad_token
    tgt_vocab['<P>'] = pad_token
    tgt_idx_vocab[pad_token] = '<P>'
    n_unique_src_words += 1
    n_unique_tgt_words += 1

    # add start and end tokens to tgt vocab (only)
    start_token = 1
    end_token = 2
    tgt_vocab['<S>'] = start_token
    tgt_vocab['<E>'] = end_token
    tgt_idx_vocab[start_token] = '<S>'
    tgt_idx_vocab[end_token] = '<E>'
    n_unique_tgt_words += 2

    # process corpus
    for sp in corpus:
        enc_input, dec_input, target = [], [], []
        dec_input.append(start_token) # append start token to dec_input
        src_seq, tgt_seq = sp
        src_words = src_seq.split(" ")
        for word in src_words:
            if word not in src_vocab.keys():
                src_vocab[word] = n_unique_src_words
                n_unique_src_words += 1
            enc_input.append(src_vocab[word])
        tgt_words = tgt_seq.split(" ")
        for word in tgt_words:
            if word not in tgt_vocab.keys():
                tgt_vocab[word] = n_unique_tgt_words
                tgt_idx_vocab[n_unique_tgt_words] = word
                n_unique_tgt_words += 1
            dec_input.append(tgt_vocab[word])
            target.append(tgt_vocab[word])
        # append end token to target
        target.append(end_token)
        dataset_item = [enc_input, dec_input, target]
        # fill padding
        for item in dataset_item:
            while len(item) < max_seq_len:
                item.append(pad_token)
        # append to dataset
        dataset.append(dataset_item)
    dataset = np.array(dataset)

    # instantiate model
    model = make_model(len(src_vocab.keys()), len(tgt_vocab.keys()), d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device).to(device)

    # instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # train loop
    for ep in tqdm(range(num_epochs)):

        # fetch minibatch
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = dataset[idx]

        # calculate loss
        loss = calculate_loss(model, minibatch, pad_token, n_heads, device)

        # update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print intermediate loss
        if ep % 100 == 0:
            print('ep:{} \t loss:{:.3f}'.format(ep, loss.item()))

    # test
    test_input = ["He is jumping on the beach .",
                  "She is running on the road .",
                  "She is climbing a mountain .",
                  "He is driving a car .",
                  "She is standing on the bed .",
                  "He is jumping on the road .",
                  "He is running on the car .",
                  "She is climbing on the beach .",
                  "She is driving on the bed .",
                  "He is standing on the mountain ."]

    print('\nTranslating test sentences:')
    model.eval()
    for sen in test_input:
        # convert input words to tokens
        words = sen.split(" ")
        enc_input = [src_vocab[w] for w in words]
        # pad
        while len(enc_input) < max_seq_len:
            enc_input.append(pad_token)
        enc_input = torch.LongTensor(enc_input).unsqueeze(0).to(device)
        enc_enc_mask = pad_mask(enc_input, pad_token, n_heads)

        ### decoding
        dec_input = [start_token]
        # pad
        while len(dec_input) < max_seq_len:
            dec_input.append(pad_token)
        dec_input = torch.LongTensor(dec_input).unsqueeze(0).to(device)

        # greedy decoding
        # dec_input = greedy_decoding(enc_input, enc_enc_mask, dec_input, max_seq_len, pad_token, n_heads)

        # beam search decoding
        dec_input = beam_search_decoding(enc_input, enc_enc_mask, dec_input, max_seq_len, beam_width, pad_token, n_heads)

        # convert translated tokens to words
        pred_words = [tgt_idx_vocab[i.item()] for i in dec_input[0]]
        print('{} -> {}'.format(words, pred_words))
