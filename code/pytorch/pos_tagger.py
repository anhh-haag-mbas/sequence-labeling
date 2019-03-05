import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


class PosTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tag_size):
        super(PosTagger, self).__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim), 
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def to_ixs(seq, ixs):
    """
    Convert words or tags (or any items in seq) to their 
    """
    try:
        out = torch.tensor([ixs[w] for w in seq], dtype=torch.long)
    except KeyError:
        out = torch.tensor([0.], dtype=torch.long)
    return out


def import_data(path):
    data = []
    with open(path, "r") as f:
        for line in f.readlines():
            if line.startswith("#"):
                words, tags = [], []
                continue
            if line.isspace():
                data.append((words, tags))
                continue

            line = line.split("\t")
            words.append(line[1])
            tags.append(line[3])

    return data



def train(model, training_data, loss_func=None, optimizer=None, epochs=5):
    remaining = len(training_data)
    for epoch in range(epochs):
        print("Testing: Epoch {}".format(epoch))
        print("Working... ", end="", flush=True)
        total_loss = 0
        for sent, tags in training_data:
            remaining -= 1
            if remaining % 50 == 0:
                print("-", end="", flush=True)

            model.zero_grad()
            model.hidden = model.init_hidden()

            out = model(to_ixs(sent, word_to_ix))
            loss = loss_func(out, to_ixs(tags, tag_to_ix))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print("Epoch loss: {}".format(total_loss))

    return model

def inspect(model, data):
    total = 0
    correct = 0
    for sent, tags in data:
        sentence = to_ixs(sent, word_to_ix)
        tag_ixs = to_ixs(tags, tag_to_ix)
        model.hidden = model.init_hidden()
        out = model(sentence)
        print("WORD\t\tTAG\t\tEXP")
        print()
        for scores, word, tag_ix in zip(out, sent, tag_ixs):
            exp_tag = ix_to_tag[tag_ix.item()]
            pred_tag = ix_to_tag[torch.argmax(scores).item()]

            total += 1
            if exp_tag == pred_tag:
                correct += 1

            print("\t\t".join([word, pred_tag, exp_tag]))
        print()
        print("{} correct of {} total. Accuracy: {}%".format(correct, total, (correct / total * 100)))
        print()




training_data = import_data("../../data/UD_Danish-DDT/da_ddt-ud-train.conllu")
test_data = import_data("../../data/UD_Danish-DDT/da_ddt-ud-test.conllu")

word_to_ix, tag_to_ix = {}, {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

ix_to_tag = {}
for tag, ix in tag_to_ix.items():
    ix_to_tag[ix] = tag

EMBEDDING_DIM = 64
HIDDEN_DIM = 64

model = PosTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_func = nn.NLLLoss()

try:
    model.load_state_dict(torch.load("./ma_model.pt"))
    model.eval()
    print("Model loaded")
except FileNotFoundError:
    train(model, training_data, loss_func, optimizer)
    torch.save(model.state_dict(), "./ma_model.pt")

test_ix = random.randint(0,len(test_data))
inspect(model, test_data[test_ix:test_ix+1])
