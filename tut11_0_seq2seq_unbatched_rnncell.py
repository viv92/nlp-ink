### Program implementing Neural Seq-to-Seq NMT using RNN on a minimal scale / corpus.

## Features:
# 1. Two RNN cells - one to encode the input sequence to a latent vector (final hidden state of encoder); and other to decode the latent vector into translated text.
# 2. The NMT dataset is organized as [input,output,target], where input is a src language sentence (input to encoder); output is the target language sentence (input to decoder) and target = output + <END> token, but the tokens in the target are indexed according to the target vocabulary (so they act as class labels for CrossEntropyLoss)
# 3. This is a simplified (and slower) implementation which processes one sequence at a time (unbatched). An advantage of unbatched implementation is that we can deal with inputs of different lengths without the need for padding. Similarly, we can deal with outputs of different lengths using the end token of each output sequence. So we dont' need to stick to a max_seq_len. Note that we still need to use a start token as the first input to the decoder.

## Todos / questions:
# 1. Do we maintain separate vocabularies for the src and target language? My guess: the goal is to have zero common embeddings between the two languages. For the case of this implementation, the desired disjoint embedding sets can be achieved by maintaining a common word_to_token dict - so that each word gets a unique integer token and hence a unique embedding. Note that num_embeddings = unique tokens in src language + unique tokens in target language, but n_classes = unique tokens in target language.
# 2. What about words that are common between the two languages but have different meanings?
# 3. handling n_classes for target during loss calculation
# 4. <END> token based termination of sequence during translation
# 5. Greedy decoding versus beam search based decoding during test / translation
# 6. LongTensor vs FloatTensor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# model class
class Seq2Seq(nn.Module):
    def __init__(self, n_unique_words, n_classes, embed_dim, h_dim, start_token, end_token_target):
        super().__init__()
        self.embedding = nn.Embedding(n_unique_words, embed_dim)
        self.enc_cell = nn.RNNCell(embed_dim, h_dim)
        self.dec_cell = nn.RNNCell(embed_dim, h_dim)
        self.fc = nn.Linear(h_dim, n_classes)
        self.start_token = start_token
        self.end_token_target = end_token_target
        self.h_dim = h_dim

    # forward prop through encoder rnn (used for both training and testing / translating)
    def encode(self, h0, input):
        # convert tokens to embeddings
        input = self.embedding(input) # input.shape: [1, input_seq_len, embed_dim]
        input_seq_len = input.shape[1]
        hn = h0
        for i in range(input_seq_len):
            x = input[:,i,:]
            hn = self.enc_cell(x, hn) # hn.shape: [1, hidden_dim]
        return hn

    # train time forward prop through decoder (used only for training as it requires ground truth decoded text as input)
    def decode_train(self, hn, output):
        # convert tokens to embeddings
        start_emb = self.embedding(torch.LongTensor([self.start_token]))
        output = self.embedding(output) # input.shape: [1, output_seq_len, embed_dim]
        output_seq_len = output.shape[1]
        pred_output = torch.empty((output_seq_len + 1, self.h_dim)) # pred_output.shape: [output_seq_len + 1, h_dim]
        hn = self.dec_cell(start_emb, hn) # first input to the decoder is the start token embedding
        pred_output[0] = hn
        for n in range(output_seq_len):
            x = output[:,n,:]
            hn = self.dec_cell(x, hn)
            pred_output[n+1] = hn
        pred_score = self.fc(pred_output) # pred_score.shape: [output_seq_len + 1, n_classes]
        return pred_score

    # test time forward prop through decoder (used only for testing / generating translation as it feedbacks output of prev step as input to curr step)
    # using greedy decoding
    def decode_test_greedy(self, hn, max_translation_len):
        # convert tokens to embeddings
        start_emb = self.embedding(torch.LongTensor([self.start_token]))
        pred_tokens = []
        x = start_emb  # first input to the decoder is the start token embedding
        for j in range(max_translation_len):
            hn = self.dec_cell(x, hn) # hn.shape: [h_dim]
            # obtain token from hn
            pred_score = self.fc(hn) # pred_score.shape: [1, n_classes]
            pred_token = torch.argmax( F.log_softmax(pred_score, dim=1), dim=1 ).squeeze().item()
            pred_tokens.append(pred_token)
            if pred_token == self.end_token_target:
                break
            # next input to decoder
            # pred_token is indexed according to target vocab, but we want input x to be indexed according to global vocab
            x = self.embedding(torch.LongTensor( [global_word_dict[target_idx_dict[pred_token]]] ))
        return pred_tokens

    # recursive beam search
    def recursive_beam_search(self, curr_beam, remaining_recursion_depth, beam_width):
        # termination case 1
        if remaining_recursion_depth == 0:
            return curr_beam
        # recursion body
        all_beam_candidates = []
        for class_token_list, token_list_score, hn in curr_beam:
            input_token = class_token_list[-1]
            # don't expand candidates for this class_token_list if the input token is <END> token
            if input_token == self.end_token_target:
                continue
            decoder_input = self.embedding(torch.LongTensor([global_word_dict[target_idx_dict[input_token]]]))
            next_hn = self.dec_cell(decoder_input, hn) # next_hn.shape: [1, h_dim]
            # obtain token from hn
            pred_scores = self.fc(next_hn) # pred_score.shape: [1, n_classes]
            pred_scores = F.log_softmax(pred_scores, dim=1).squeeze(0)
            # new beam candidate: triplet [ new_class_token_list, new_token_list_score, next_hn ]
            for class_token, score in enumerate(pred_scores):
                new_class_token_list = class_token_list + [class_token]
                new_token_list_score = ( token_list_score * len(class_token_list) + score ) / ( len(class_token_list) + 1 ) # length normalized score
                new_beam_candidate = [ new_class_token_list, new_token_list_score, next_hn ]
                # append to list of all candidates
                all_beam_candidates.append(new_beam_candidate)
        # termination case 2
        if all_beam_candidates == []:
            return curr_beam
        # select topk candidates
        all_beam_candidates = list(sorted(all_beam_candidates, key=lambda x: x[1], reverse=True))
        updated_beam = all_beam_candidates[:beam_width]
        # recurse
        final_beam = self.recursive_beam_search(updated_beam, remaining_recursion_depth-1, beam_width)
        return final_beam

    # test time forward prop through decoder (used only for testing / generating translation as it feedbacks output of prev step as input to curr step)
    # using beam search
    def decode_test_beamSearch(self, hn, max_translation_len, beam_width):
        # convert tokens to embeddings
        start_emb = self.embedding(torch.LongTensor([self.start_token])) # start_emb.shape: [1, embed_dim]
        next_hn = self.dec_cell(start_emb, hn) # next_hn.shape: [1, h_dim]
        # obtain tokens and next inputs from hn
        pred_scores = self.fc(next_hn) # pred_score.shape: [1, n_classes]
        pred_scores = F.log_softmax(pred_scores, dim=1).squeeze(0)
        # beam candidates: list of triplets [ class_token_list, token_list_score, next_hn ]
        beam_candidates = [ [ [class_token], score.item(), next_hn ] for class_token, score in enumerate(pred_scores) ]
        # select topk candidates
        beam_candidates = list(sorted(beam_candidates, key=lambda x: x[1], reverse=True))
        initial_beam = beam_candidates[:beam_width]
        # recursive beam search
        final_beam = self.recursive_beam_search(initial_beam, max_translation_len-1, beam_width)
        # pick out the best class_token_list
        best_class_token_list, best_score, final_hn = final_beam[0]
        return best_class_token_list

    # function to forward prop a sequence of words through the encoder and the decoder
    def forward(self, h0, input, output):
        hn = self.encode(h0, input)
        pred_output = self.decode_train(hn, output)
        return pred_output

    # returns word embedding
    def get_embedding(self, x):
        emb = self.embedding(x)
        return emb


# function to calculate cross entropy loss
def calculate_loss(model, h0, input, output, target):
    input, output, target = torch.LongTensor(input), torch.LongTensor(output), torch.tensor(target)
    scores = model(h0.unsqueeze(0), input.unsqueeze(0), output.unsqueeze(0)) # scores.shape: [output_seq_len + 1, n_classes] - Note that CE_loss will interpret seq dimension as batch dimension
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss = criterion(scores, target)
    return loss


# main
if __name__ == '__main__':

    # hyperparams
    embed_dim = 5
    h_dim = 10
    lr = 1e-2
    num_epochs = 5000
    max_translation_len = 10 # to avoid infinite length translations
    beam_width = 3 # for beam search during translation
    random_seed = 42

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

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

    ### create word_dict and NMT dataset
    nmt_dataset = []
    n_unique_global_words, n_unique_target_words = 0, 0
    global_word_dict, global_idx_dict, target_word_dict, target_idx_dict = {}, {}, {}, {}
    # add start and end token
    start_token = 0
    end_token_global = 1 # end token according to global vocab
    end_token_target = 0 # end token according to target vocab
    global_word_dict['<START>'] = start_token
    global_word_dict['<END>'] = end_token_global
    global_idx_dict[start_token] = '<START>'
    global_idx_dict[end_token_global] = '<END>'
    target_word_dict['<END>'] = end_token_target
    target_idx_dict[end_token_target] = '<END>'
    n_unique_global_words += 2
    n_unique_target_words += 1
    # process corpus
    for sp in corpus:
        nmt_input, nmt_output, nmt_target = [], [], []
        input, output = sp
        input_words = input.split(" ")
        for word in input_words:
            if word not in global_word_dict.keys():
                global_word_dict[word] = n_unique_global_words
                global_idx_dict[n_unique_global_words] = word
                n_unique_global_words += 1
            nmt_input.append(global_word_dict[word])
        output_words = output.split(" ")
        for word in output_words:
            if word not in global_word_dict.keys():
                global_word_dict[word] = n_unique_global_words
                global_idx_dict[n_unique_global_words] = word
                n_unique_global_words += 1
            if word not in target_word_dict.keys():
                target_word_dict[word] = n_unique_target_words
                target_idx_dict[n_unique_target_words] = word
                n_unique_target_words += 1
            nmt_output.append(global_word_dict[word])
            nmt_target.append(target_word_dict[word])
        # append <END> token to target
        nmt_target.append(end_token_target)
        # append [input, output, target] tuple to dataset
        nmt_dataset.append([nmt_input, nmt_output, nmt_target])

    # instantiate model
    model = Seq2Seq(n_unique_global_words, n_unique_target_words, embed_dim, h_dim, start_token, end_token_target)

    # instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # train loop
    for ep in range(num_epochs):
        epoch_loss = 0

        for train_item in nmt_dataset:
            input, output, target = train_item

            # calculate loss
            h0 = torch.zeros(h_dim) # first hidden state of encoder
            loss = calculate_loss(model, h0, input, output, target)
            epoch_loss += loss

        # update params
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()

        # print intermediate loss
        if ep % 1000 == 0:
            print('ep:{} \t loss:{:.3f}'.format(ep, epoch_loss.item()))


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
    for sen in test_input:
        # convert input words to tokens
        words = sen.split(" ")
        nmt_input = torch.LongTensor([global_word_dict[w] for w in words])
        # get translated tokens from model
        h0 = torch.zeros(h_dim)
        hn = model.encode(h0.unsqueeze(0), nmt_input.unsqueeze(0))
        pred_tokens = model.decode_test_beamSearch(hn, max_translation_len, beam_width) # note that the predicted tokens are indexed according to target vocab
        # convert translated tokens to words
        pred_words = [target_idx_dict[i] for i in pred_tokens]
        print('{} -> {}'.format(words, pred_words))
