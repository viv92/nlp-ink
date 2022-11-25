### Program implementing Neural Network Language Model (NNLM) using an bi-GRU

## Features:
# 1. GRU allows processing of input sequence of different lengths / num of words (though, for implementation, we need to fix on a max seq len)
# 2. target word sequence is one right shifted version of the input word sequence

## Todos / questions:
# 1.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# model class
class GRULM(nn.Module):
    def __init__(self, n_classes, embed_dim, h_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.gru = nn.GRU(embed_dim, h_dim, bidirectional=True)
        self.fc = nn.Linear(h_dim*2, n_classes)

    # function to forward prop a sequence of words
    def forward(self, inputs, h0): # inputs.shape: [batch_size, seq_len]; h0.shape: [2, batch_size, h_dim]
        x = self.embedding(inputs) # x.shape: [batch_size, seq_len, embed_dim]
        x = x.transpose(0, 1) # x.shape: [seq_len, batch_size, embed_dim]
        out, hn = self.gru(x, h0) # out.shape: [seq_len, batch_size, h_dim*2]; hn.shape: [2, batch_size, h_dim]
        scores = self.fc(out) # scores.shape: [seq_len, batch_size, n_classes]
        scores = scores.transpose(0,1) # scores.shape: [batch_size, seq_len, n_classes]
        scores = scores.transpose(1,2) # scores.shape: [batch_size, n_classes, seq_len]
        return scores, hn

    # returns word embedding
    def get_embedding(self, x):
        emb = self.embedding(x)
        return emb


# function to extract random minibatch
def get_minibatch(dataset, batch_size):
    N = len(dataset)
    indices = np.arange(N)
    np.random.shuffle(indices)
    batch_indices = indices[:batch_size]
    inputs = [dataset[i][0] for i in batch_indices]
    targets = [dataset[i][1] for i in batch_indices]
    inputs = torch.LongTensor(inputs)
    targets = torch.tensor(targets)
    return inputs, targets


# function to calculate cross entropy loss over minibatch
def calculate_loss(model, inputs, targets, h0):
    scores, hn = model(inputs, h0) # outputs.shape: [batch_size, n_classes, seq_len]
    criterion = nn.CrossEntropyLoss()
    loss = criterion(scores, targets)
    return loss


# # visualize learnt embeddings on a 2-d plane (dimension reduced using PCA)
# def visualize_embeddings(word_dict, model, vocab_size, embed_dim):
#     inputs = np.arange(vocab_size) # idx of all words in vocab
#     inputs = torch.FloatTensor(inputs)
#     embeddings = model.get_embedding(inputs.unsqueeze(1)) # shape: [vocab_size, embed_dim]
#     # reduce embedding dimension to 2
#     U,S,V = torch.pca_lowrank(embeddings) # note that this centers the embeddings matrix internally
#     embeddings_reduced = torch.matmul(embeddings, V[:, :2]) # shape: [vocab_size, 2]
#     # plot
#     for word, idx in word_dict.items():
#         x,y = embeddings_reduced[idx].data.numpy()
#         plt.scatter(x, y)
#         plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
#     plt.show()



# main
if __name__ == '__main__':

    # hyperparams
    seq_len = 6 # max seq len for the input
    embed_dim = 5
    h_dim = 5
    batch_size = 8
    lr = 1e-2
    num_epochs = 5000
    random_seed = 42

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # given corpus
    corpus = ["He likes fruits", "He hates vegetables", "She hates fruits", "She likes vegetables", "She likes brocoli", "She hates orange", "He likes apple", "He hates cabagge"]

    # create vocab and word dict
    corpus_string = " ".join(corpus)
    word_list = corpus_string.split(' ')
    vocab = list(set(word_list))
    vocab.sort() # to get a deterministic ordering of words
    vocab_size = len(vocab) # n_classes
    word_dict = {w:i for i,w in enumerate(vocab)} # encoding word to idx - used for training
    index_dict = {i:w for i,w in enumerate(vocab)} # encoding idx to word - used for prediction

    # create nnlm dataset
    # inputs are sequences of words each of seq_len
    # targets are same as inputs but one step right-shifted
    nnlm_dataset = []
    for i in range(seq_len, len(word_list)):
        target_seq_idx = []
        input_seq_idx = []
        for j in range(i-seq_len, i):
            input_seq_idx.append(word_dict[word_list[j]])
            target_seq_idx.append(word_dict[word_list[j+1]])
        nnlm_dataset.append([input_seq_idx, target_seq_idx])

    # instantiate model
    model = GRULM(vocab_size, embed_dim, h_dim)

    # instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # train loop
    for ep in range(num_epochs):

        # get random minibatch
        # inputs.shape: [batch_size, seq_len]
        # targets.shape: [batch_size, seq_len]
        inputs, targets = get_minibatch(nnlm_dataset, batch_size)

        # calculate loss
        h0 = torch.zeros(2, batch_size, h_dim) # starting hidden state of GRU
        loss = calculate_loss(model, inputs, targets, h0)

        # update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print intermediate loss
        if ep % 1000 == 0:
            print('ep:{} \t loss:{:.3f}'.format(ep, loss.item()))

    # predict (abusing train data as test data)
    print('\nPredicting the next word given a sequence of words as input:')
    for i in range(seq_len, len(word_list)):
        input_seq_idx = []
        prev_words = []
        for j in range(i-seq_len, i):
            prev_words.append(word_list[j])
            input_seq_idx.append(word_dict[word_list[j]])
        input_seq = torch.LongTensor(input_seq_idx)
        h0 = torch.zeros(2, 1, h_dim) # starting hidden state of GRU
        out, hn = model(input_seq.unsqueeze(0), h0) # out.shape: [1, n_classes, seq_len]
        out = out.squeeze(0)
        predicted_classes = torch.argmax(F.log_softmax(out, dim=0), dim=0).detach().numpy()
        predicted_words = [index_dict[x] for x in predicted_classes]
        for k in range(seq_len):
            print('{} -> {}'.format(prev_words[:k+1], predicted_words[k]))

    # generate
    print('\nGenerating a sequence of words that follows the given input sequence:')
    for i in range(seq_len, len(word_list)):
        input_seq_idx = []
        prev_words = []
        for j in range(i-seq_len, i):
            prev_words.append(word_list[j])
            input_seq_idx.append(word_dict[word_list[j]])
        input_seq = torch.LongTensor(input_seq_idx)
        h0 = torch.zeros(2, 1, h_dim) # starting hidden state of GRU
        out, hn = model(input_seq.unsqueeze(0), h0) # out.shape: [1, n_classes, seq_len]; hn.shape: [2, 1, h_dim]
        # extract the final output
        out = out[:,:,-1]

        # use the final output and hidden state to generate further text
        generated_words = []
        for k in range(seq_len):
            out = out.squeeze(0)
            predicted_class = torch.argmax(F.log_softmax(out, dim=0), dim=0).item()
            predicted_word = index_dict[predicted_class]
            generated_words.append(predicted_word)
            next_input = torch.LongTensor([[predicted_class]])
            out, hn = model(next_input, hn)
        print('{} -> {}'.format(prev_words, generated_words))
