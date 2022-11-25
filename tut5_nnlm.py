### Program implementing Neural Network Language Model (NNLM): [https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf]

## Features:
# 1. Language models predict the next word given a sequence of words. This is a simplified language model in that the input sequence of words has a fixed (predefined) length.
# 2. We use a resnet-style two layer network to represent the language model

## Todos / questions:
# 1.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# model class
class NNLM(nn.Module):
    def __init__(self, n_steps, n_classes, embed_dim, h_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.h_1 = nn.Linear(n_steps*embed_dim, h_dim, bias=False)
        self.b_1 = nn.Parameter(torch.ones(h_dim))
        self.h_21 = nn.Linear(h_dim, n_classes, bias=False)
        self.h_22 = nn.Linear(n_steps*embed_dim, n_classes, bias=False)
        self.b_2 = nn.Parameter(torch.ones(n_classes))

    def forward(self, inputs): # inputs.shape: [batch_size, n_steps]
        x = self.embedding(inputs) # x.shape: [batch_size, n_steps, embed_dim]
        # flatten last two dimensions
        x = x.flatten(start_dim=1, end_dim=2) # x.shape: [batch_size, n_steps * embed_dim]
        h = F.tanh(self.h_1(x) + self.b_1)
        out = self.h_21(h) + self.h_22(x) + self.b_2 # out is the raw score vector of shape [batch_size, n_classes]
        return out

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
def calculate_loss(model, inputs, targets):
    outputs = model(inputs) # outputs.shape: [batch_size, n_classes]
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
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
    n_steps = 2 # number of words in input subsequence
    embed_dim = 5
    h_dim = 5
    batch_size = 8
    lr = 1e-3
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
    nnlm_dataset = []
    for i in range(n_steps, len(word_list)):
        target_word_idx = word_dict[word_list[i]]
        prev_words_idx_list = []
        for j in range(i-n_steps, i):
            prev_words_idx_list.append(word_dict[word_list[j]])
        nnlm_dataset.append([prev_words_idx_list, target_word_idx])

    # instantiate model
    model = NNLM(n_steps, vocab_size, embed_dim, h_dim)

    # instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # train loop
    for ep in range(num_epochs):

        # get random minibatch
        # inputs.shape: [batch_size, n_steps]
        # targets.shape: [batch_size]
        inputs, targets = get_minibatch(nnlm_dataset, batch_size)

        # calculate loss
        loss = calculate_loss(model, inputs, targets)

        # update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print intermediate loss
        if ep % 1000 == 0:
            print('ep:{} \t loss:{:.3f}'.format(ep, loss.item()))

    # predict (abusing train data as test data)
    print('Predicting the next word:')
    for i in range(n_steps, len(word_list)):
        prev_words_idx_list = []
        prev_words = ""
        for j in range(i-n_steps, i):
            prev_words += " " + word_list[j]
            prev_words_idx_list.append(word_dict[word_list[j]])
        input_seq = torch.LongTensor(prev_words_idx_list)
        out = model(input_seq.unsqueeze(0))
        predicted_class = torch.argmax(F.log_softmax(out, dim=1)).item()
        predicted_word = index_dict[predicted_class]
        print('{} -> {}'.format(prev_words, predicted_word))
