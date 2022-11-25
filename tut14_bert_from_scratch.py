### Program implementing BERT from scratch in pytorch

## Features:
# 1. Obtaining a language model using masked language modelling (MLM) and next sentence prediction (NSP) tasks, that transfers well to downstream tasks via finetuning.
# 2. No [START] and [END] tokens in bert. Instead we have [CLS], [SEP], [PAD] and [MASK] tokens.

## Todos / Questions:
# 1. prepare masked input sentence pairs
# 2. input embedding = token embedding + (learnt) position embedding + segment embedding
# 3. prepare target vectors for MLM and NSP tasks
# 4. add an output head (effective decoder) to transformer encoder to get predicted vectors for both MLM and NSP tasks
# 5. check correctness of masked_position = -1 for padded masked_positions
# 6. replace layernorm with nn.layernorm
# 7. do we need max_pred at all?
# 8. does multiplying embeddings with math.sqrt(d_model) create mismatch when inverting the output to embeddings in the decoder head? 

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

# utility function for masking input
def mask_input(input, word_dict, max_pred):
    mask_token = word_dict['[MASK]']
    n = len(input)
    n = min(max_pred, max(int(.15 * n), 1) ) # number of tokens that will be masked: 1 <= 15% of len(input) <= max_pred
    masked_positions = list( np.random.choice(np.arange(len(input)), n, replace=False) )
    masked_tokens = [input[idx] for idx in masked_positions]
    # apply mask
    for pos in masked_positions:
        r = np.random.choice(10)
        if r <= 8: # 80 %
            input[pos] = word_dict['[MASK]']
        if r == 9: # 10%
            input[pos] = np.random.choice(len(word_dict.keys()))
    return input, masked_positions, masked_tokens

# utility function to create mask over pad tokens
def pad_mask(keys, pad_token):
    batch_size, max_seq_len = keys.shape
    mask = keys.eq(pad_token).unsqueeze(1) # mask.shape: [batch_size, 1, max_seq_len]
    mask = mask.expand(batch_size, max_seq_len, max_seq_len) # mask.shape: [batch_size, max_seq_len, max_seq_len]
    return mask == 0 # False elements are masked

# utility function to calculate train loss (cross entropy loss)
def calculate_loss(model, minibatch_col1, minibatch_col2, minibatch_col3, minibatch_col4, word_dict, device):
    input = torch.LongTensor(minibatch_col1).to(device)
    masked_positions = torch.LongTensor(minibatch_col2).to(device)
    masked_tokens = torch.LongTensor(minibatch_col3).to(device)
    is_next_label = torch.LongTensor(minibatch_col4).to(device)

    # obtain pad mask for input
    # pad masking is applied only on the basis of keys (no need to refer queries - this works out because keys = values in attention)
    pad_token = word_dict['[PAD]']
    input_mask = pad_mask(input, pad_token)

    # forward prop through model
    sep_token = word_dict['[SEP]']
    nsp_scores, mlm_scores = model(input, input_mask, masked_positions, sep_token) # nsp_scores.shape: [batch_size, 2]; mlm_scores.shape: [batch_size, max_pred, vocab_size]
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # nsp loss
    nsp_loss = criterion(nsp_scores, is_next_label)
    # mlm loss
    mlm_scores = mlm_scores.transpose(1, 2) # scores.shape: [batch_size, vocab_size, max_pred]
    mlm_loss = criterion(mlm_scores, masked_tokens)
    return nsp_loss + mlm_loss

# class to obtain embeddings from vocab tokens; embed_dim = d_model
# note that for BERT, embedding = token_emb + position_emb + segment_emb
class Embeddings(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, device):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.seg_emb = nn.Embedding(2, d_model) # since only two segments for each input
        self.d_model = d_model
        self.device = device
    def forward(self, x, sep_token):
        # token embeddings
        token_emb = self.token_emb(x)
        # build positional embeddings
        batch_size, max_seq_len = x.shape
        positions = torch.arange(max_seq_len).to(self.device)
        positions = positions.unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        pos_emb = self.pos_emb(positions)
        # build segment embeddings
        segment_tokens = torch.zeros(x.shape, dtype=torch.int).to(self.device)
        idx = torch.where(x == sep_token)
        for r,c in zip(idx[0], idx[1]):
            segment_tokens[r,c:] = 1
        seg_emb = self.seg_emb(segment_tokens)
        # final embeddings
        final_emb = token_emb + pos_emb + seg_emb
        return final_emb * math.sqrt(self.d_model) # shape: [batch_size, max_seq_len, d_model]

# class implementing the feed forward block (used for each encoder / decoder layer - after the multihead attention block)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w2(self.dropout(self.w1(x).relu()))

# class implementing multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.output = nn.Linear(d_v * n_heads, d_model, bias=False)
        self.attn_weights = None # placeholder to store attention weights (used to visualize attention matrices)
        self.dropout = nn.Dropout(dropout)

    # function to calculate (masked or unmasked) multihead attention
    def forward(self, key, query, value, mask=None): # can be used for both (unmasked) encoder attention and (masked) decoder attention
        # key.shape: [batch_size, seq_len, d_model]; mask.shape: [batch_size, seq_len, seq_len]
        # project key, query, value and reshape into multiple heads
        batch_size = key.shape[0]
        # (batch_size, seq_len, d_model) -proj-> (batch_size, seq_len, proj_dim) -view-> (batch_size, seq_len, n_heads, d_k) -transpose-> (batch_size, n_heads, seq_len, d_k)
        proj_key = self.W_K(key).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)
        proj_query = self.W_Q(query).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)
        proj_value = self.W_V(value).view(batch_size, -1, self.n_heads, d_v).transpose(1, 2)
        # expand mask for n_heads
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # mask.shape: [batch_size, n_heads, seq_len, seq_len]
        # calculate attention
        attn_multihead, attn_weights = self.scaled_dotprod_attn(proj_key, proj_query, proj_value, mask, self.dropout)
        attn_multihead = attn_multihead.transpose(1, 2) # attn_multihead.shape: [batch_size, seq_len, n_heads, d_v]
        attn_multihead = torch.flatten(attn_multihead, start_dim=-2, end_dim=-1) # attn_multihead.shape: [batch_size, seq_len, n_heads * d_v]
        attn_multihead = self.output(attn_multihead) # attn_multihead.shape: [batch_size, seq_len, d_model]
        self.attn_weights = attn_weights
        return attn_multihead

    # function to calculate scaled dot product attention for one head
    def scaled_dotprod_attn(self, key, query, value, mask=None, dropout=None): # key.shape: [batch_size, seq_len, proj_dim]
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k) # attn_scores.shape: [batch_size, n_heads, seq_len, seq_len]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
        attn_weights = attn_scores.softmax(dim=-1) # attn_weights.shape: [batch_size, n_heads, seq_len, seq_len]
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        attn_vector = torch.matmul(attn_weights, value) # attn_vector.shape: [batch_size, n_heads, seq_len, d_v]
        return attn_vector, attn_weights

# class implementing Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * ( (x - mean) / (std + self.eps) ) + self.b

# class implementing residual + normalization connection - takes in any block and applies a residual connection around it + a layer normalization on top
class SublayerConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer): # sublayer can be any functional block
        return self.norm( x + self.dropout(sublayer(x)) )

# class implementing a single encoder layer
# each encoder layer has two blocks: 1. (self) multihead attention 2. feed_forward; with sublayer connection around each
class EncoderLayer(nn.Module):
    def __init__(self, self_attn, feed_forward, dim, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(dim, dropout), 2) # one for self_attn block and other for feed_forward block
    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask)) # x.shape: [batch_size, seq_len, d_model]
        x = self.sublayers[1](x, self.feed_forward) # x.shape: [batch_size, seq_len, d_model]
        return x

# class implementing the entire encoder block = stacked encoder layers
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# class implemeting the BERT = encoder + custom decoder output head
class BERT(nn.Module):
    def __init__(self, encoder, f_emb, vocab_size, d_model, device):
        super().__init__()
        self.encoder = encoder
        self.f_emb = f_emb
        self.d_model = d_model
        self.proj_nsp = nn.Linear(d_model, vocab_size, bias=False)
        self.proj_mlm = f_emb.token_emb.weight.T # shape: [d_model, vocab_size]
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.device = device
    def encode(self, input, input_mask, sep_token):
        encoder_out = self.encoder(self.f_emb(input, sep_token), input_mask)
        return encoder_out
    def forward(self, input, input_mask, masked_positions, sep_token):
        # pass through encoder
        encoder_out = self.encode(input, input_mask, sep_token) # encoder_out.shape: [batch_size, seq_len, d_model]
        # get scores for NSP
        nsp_scores = self.proj_nsp(encoder_out[:, 0]) # nsp_scores.shape: [batch_size, vocab_size]
        # get scores for MLM
        masked_positions = masked_positions.unsqueeze(-1).expand(-1, -1, self.d_model) # masked_positions.shape: [batch_size, max_pred, d_model]
        encoder_preds = torch.gather(encoder_out, dim=1, index=masked_positions) # encoder_preds.shape: [batch_size, max_pred, d_model]
        mlm_scores = torch.matmul(encoder_preds, self.proj_mlm.to(self.device)) # mlm_scores.shape: [batch_size, max_pred, vocab_size]
        return nsp_scores, mlm_scores
    # function used for test time generation - outputs score only for last element in seq
    def generate(self):
        pass


# caller function to instantiate the BERT model, using the defined hyperparams as input
def make_model(vocab_size, max_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device):
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    f_emb = Embeddings(vocab_size, max_seq_len, d_model, device) # function to convert tokens into embeddings
    encoder_layer = EncoderLayer(deepcopy(attn), deepcopy(ff), d_model, dropout) # single encoder layer
    encoder = Encoder(encoder_layer, n_layers) # encoder = stacked encoder layers
    model = BERT(encoder, f_emb, vocab_size, d_model, device)

    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


# utility function implementing greedy decoding
def greedy_decoding(encoder_out, enc_enc_mask, dec_input, max_seq_len, pad_token):
    for i in range(max_seq_len - 1):
        dec_dec_mask = pad_mask(dec_input, pad_token)
        dec_dec_mask = torch.logical_and(dec_dec_mask, subsequent_mask(dec_input.shape).to(device))
        decoder_out = model.decode(dec_input, encoder_out, enc_enc_mask, dec_dec_mask)
        score = model.generate(decoder_out, i) # score.shape: [1, tgt_vocab_size]
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
def recursive_beam_search(curr_beam, encoder_out, recursion_depth, max_seq_len, beam_width, pad_token):
    all_beam_candidates = []
    # termination case
    if recursion_depth == max_seq_len:
        return curr_beam

    for dec_input, logprob in curr_beam:
        dec_dec_mask = pad_mask(dec_input, pad_token)
        dec_dec_mask = torch.logical_and(dec_dec_mask, subsequent_mask(dec_input.shape).to(device))
        decoder_out = model.decode(dec_input, encoder_out, enc_enc_mask, dec_dec_mask)
        score = model.generate(decoder_out, recursion_depth-1) # score.shape: [1, tgt_vocab_size]
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
    final_beam = recursive_beam_search(new_beam, encoder_out, recursion_depth+1, max_seq_len, beam_width, pad_token)
    return final_beam


# utility function implementing beam search decoding
def beam_search_decoding(encoder_out, dec_input, max_seq_len, beam_width, pad_token):
    all_beam_candidates = []
    recursion_depth = 1

    dec_dec_mask = pad_mask(dec_input, pad_token)
    dec_dec_mask = torch.logical_and(dec_dec_mask, subsequent_mask(dec_input.shape).to(device))
    decoder_out = model.decode(dec_input, encoder_out, enc_enc_mask, dec_dec_mask)
    score = model.generate(decoder_out, recursion_depth-1) # score.shape: [1, tgt_vocab_size]
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
    final_beam = recursive_beam_search(initial_beam, encoder_out, recursion_depth+1, max_seq_len, beam_width, pad_token)
    # pick out the best class_token_list
    best_dec_input, best_logprob = final_beam[0]
    return best_dec_input




# main
if __name__ == '__main__':
    # hyperparams
    d_model = 512
    d_k = 64
    d_v = 64
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_pred = 5 # maximum tokens to be masked in a sequence = dim of prediction vector in decoder out
    max_seq_len = 30 # useful for padding and recursion depth in beam search based translation
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

    # given (sequential) corpus for MLM and NSP tasks
    corpus = ['Hello , how are you ? I am Romeo .',
            'Hello , Romeo My name is Juliet . Nice to meet you .',
            'Nice meet you too . How are you today ?',
            'Great . My baseball team won the competition .',
            'Oh Congratulations , Juliet .',
            'Thank you Romeo .']

    ## create vocabulary / word_dict
    corpus_string = " ".join(corpus)
    corpus_words = corpus_string.split(' ')
    vocab = list(set(corpus_words))
    vocab.sort() # sort to get a deterministic ordering of keys in word_dict
    # add special tokens to vocab
    vocab = ['[PAD]', '[CLS]', '[SEP]', '[MASK]'] + vocab
    # create word_dict
    word_dict = {w:i for i,w in enumerate(vocab)}

    ## create dataset: [input, masked_positions, masked_tokens, is_next_label]
    # input format: [[CLS] + sentence1 + [SEP] + sentence2]
    dataset_col1, dataset_col2, dataset_col3, dataset_col4 = [],[],[],[]
    for i in range(len(corpus)):
        sen1 = corpus[i]
        for j in range(len(corpus)):
            sen2 = corpus[j]
            input = '[CLS] ' + sen1 + ' [SEP] ' + sen2
            # tokenize
            input = input.split(' ')
            input = [word_dict[w] for w in input]
            # add masking
            input, masked_positions, masked_tokens = mask_input(input, word_dict, max_pred)
            # add padding
            while len(input) < max_seq_len:
                input.append(word_dict['[PAD]'])
            while len(masked_tokens) < max_pred:
                masked_positions.append(max_seq_len - 1)
                masked_tokens.append(input[max_seq_len - 1])
            # is_next_label
            is_next_label = int(j == i+1)
            # append to dataset
            dataset_col1.append(input)
            dataset_col2.append(masked_positions)
            dataset_col3.append(masked_tokens)
            dataset_col4.append(is_next_label)

    # instantiate model
    model = make_model(len(word_dict.keys()), max_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, device).to(device)

    # instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # train loop
    for ep in tqdm(range(num_epochs)):

        # fetch minibatch
        idx = np.arange(len(dataset_col1))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch_col1 = [dataset_col1[i] for i in idx]
        minibatch_col2 = [dataset_col2[i] for i in idx]
        minibatch_col3 = [dataset_col3[i] for i in idx]
        minibatch_col4 = [dataset_col4[i] for i in idx]

        # calculate loss
        loss = calculate_loss(model, minibatch_col1, minibatch_col2, minibatch_col3, minibatch_col4, word_dict, device)

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
        enc_enc_mask = pad_mask(enc_input, pad_token)
        encoder_out = model.encode(enc_input, enc_enc_mask) # encoder_out.shape: [1, max_seq_len, d_model]

        ### decoding
        dec_input = [start_token]
        # pad
        while len(dec_input) < max_seq_len:
            dec_input.append(pad_token)
        dec_input = torch.LongTensor(dec_input).unsqueeze(0).to(device)

        # greedy decoding
        # dec_input = greedy_decoding(encoder_out, enc_enc_mask, dec_input, max_seq_len, pad_token)

        # beam search decoding
        dec_input = beam_search_decoding(encoder_out, dec_input, max_seq_len, beam_width, pad_token)

        # convert translated tokens to words
        pred_words = [tgt_idx_vocab[i.item()] for i in dec_input[0]]
        print('{} -> {}'.format(words, pred_words))
