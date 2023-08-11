from torch import nn
import time
import torch


def train(dataloader, model, optimizer, criterion, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
        start_time = time.time()
        return total_acc / total_count


def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0
    trues = []
    preds = []
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            trues.extend(label)
            preds.extend(predicted_label.argmax(1))
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count, {"True": trues, "Predicted":preds}

def deep_evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0
    trues = []
    preds = []
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count, {}



class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim = 64, num_class = 5):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class * 2)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Linear( num_class * 2, num_class)
        self.relu2 = nn.ReLU(True)
        self.fc3 = nn.Linear( num_class, num_class // 2)
        self.fc4 = nn.Linear( num_class // 2, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        fully1 = self.fc(embedded)
        relu = self.relu(fully1)
        fully2 = self.fc2(relu)
        relu2 = self.relu2(fully2)
        fully3 = self.fc3(relu2)
        fully4 = self.fc4(fully3)
        return fully4