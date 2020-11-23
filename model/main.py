import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torchtext.data import Field, BucketIterator, TabularDataset
from py.encoder_module import Encoder
from py.decoder_module import Decoder
from py.attention_module import Attention
from py.seq2seq import S2S


def data_process(max_data_len, train_data_path, vocab_min_freq, test_data_path):
    datasets = Field(truncate_first=max_data_len, fix_length=max_data_len)
    train_data = TabularDataset(path=train_data_path, format='csv',
                                fields=[('theme', datasets), ('keyword', datasets),
                                        ('src', datasets), ('tgt', datasets)],
                                skip_header=True)
    datasets.build_vocab(train_data, min_freq=vocab_min_freq)
    data_size = len(datasets.vocab)
    train_dataloader = BucketIterator(train_data, batch_size=BATCH_SIZE, device=device, shuffle=True)

    test_data = TabularDataset(path=test_data_path, format='csv',
                               fields=[('theme', DATA), ('keyword', DATA), ('src', DATA), ('tgt', DATA)],
                               skip_header=True)
    test_dataloader = BucketIterator(test_data, batch_size=BATCH_SIZE, device=device, shuffle=False)

    return datasets, data_size, train_dataloader, test_dataloader


def init_weights(m: nn.Module):
    # parameter initialization

    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def train(model, dataset, optimizer, criterion, clip, teach):

    model.train()
    epoch_loss = 0

    for _, batch in enumerate(dataset):
        cov_loss = 0
        theme = batch.theme[0:THEME_LEN, :]  # (batch,1)
        keyword = batch.keyword[0:KEYWORD_LEN, :]  # (batch,1)
        src = batch.src[0:SRC_LEN, :]  # (batch, s_len)
        tgt = batch.tgt[0:TGT_LEN, :]  # (batch, t_len)

        optimizer.zero_grad()

        outputs, attns = model(theme, keyword, src, tgt, teach)

        cov = attns.get("coverage", None)
        std = attns.get("std", None)
        for j in range(len(cov)):
            cov_loss += torch.min(std[j], cov[j]).sum()
        cov_loss *= LAMBDA_COVERAGE
        outputs = outputs.contiguous()
        outputs = outputs.view(-1, DATA_SIZE)
        tgt = tgt.contiguous()
        tgt = tgt[1:].view(-1)
        loss = criterion(outputs, tgt)
        loss = loss + cov_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataset)


def eval(model, dataset):

    model.eval()

    teach = 0
    for _, batch in enumerate(dataset):
        theme = batch.theme[0:THEME_LEN, :]  # (batch,1)
        keyword = batch.keyword[0:KEYWORD_LEN, :]  # (batch,1)
        src = batch.src[0:SRC_LEN, :]  # (batch, s_len)
        tgt = batch.tgt[0:TGT_LEN, :]  # (batch, t_len)
        outputs, attns = model(theme, keyword, src, tgt, teach)
        outputs = outputs.contiguous()
        texts = outputs.max(2)[1]
    return texts


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def iters(model, criterion,  learning_rate, load_from_checkpoint, epochs, train_data, clip, teach_rate,
          checkpoint_times, model_save_path, para_save_path, checkpoint_path=None):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, threshold=0.0001,
                                                     threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-9,
                                                     verbose=False)

    plot_loss = []
    start_epoch = 0

    if load_from_checkpoint:
        path_checkpoint = checkpoint_path  # checkpoint path
        checkpoint = torch.load(path_checkpoint)  # load checkpoint
        model.load_state_dict(checkpoint['net'])  # load parameter

        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']  # set start epoch
        scheduler.load_state_dict(checkpoint['scheduler'])
        plot_loss = checkpoint['loss']

    for epoch in range(start_epoch, epochs):
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(optimizer.param_groups[-1]['lr'])
        start_time = time.time()

        train_loss = train(model, train_data, optimizer, criterion, clip, teach_rate)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.2f}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

        plot_loss.append(train_loss)

        scheduler.step(train_loss)

        if epoch > 0 and epoch % checkpoint_times == 0:
            # checkpoint
            model_name = f'{model_save_path}/model_{epoch}.pt'
            para_name = f'{para_save_path}/para_{epoch}.pt'
            torch.save(model, model_name)
            checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "scheduler": scheduler.state_dict(),
                "loss": plot_loss}
            torch.save(checkpoint, para_name)


"""
data should be csv.
with 4 col ['theme', 'keyword', 'src', 'tgt']
length of 'theme' must be 1
"""
device = torch.device('cuda:0')
BATCH_SIZE = 64
TEST_BATCH_SIZE = 10
HIDDEN_SIZE = 512
EPOCHS = 101
CLIP = 1
LEARNING_RATE = 3e-5
ENC_DROPOUT = 0.2  # if use, please change in Encoder and Decoder
DEC_DROPOUT = 0.2  # if use, please change in Encoder and Decoder
TEACH_RATE = 1
LAMBDA_COVERAGE = 1
MAX_DATA_LEN = 80
THEME_LEN = 1  # fixed set as 1
KEYWORD_LEN = 1
SRC_LEN = 80
TGT_LEN = 80
VOCAB_MIN_FREQUENCY = 1
GENERATION_LEN = 80
CHECKPOINT_TIMES = 10  # every XX times save the model and parameters
TRAIN_DATA_PATH = '...'  # path of train data
TEST_DATA_PATH = '...'  # path of test data
MODEL_SAVE_PATH = '...'
PARA_SAVE_PATH = '...'
LOAD_FROM_CHECKPOINT = False  # if True, please set checkpoint path
CHECKPOINT_PATH = None  # path of checkpoint

DATA, DATA_SIZE, train_dataset, test_dataset = data_process(MAX_DATA_LEN, TRAIN_DATA_PATH,
                                                            TEST_DATA_PATH, VOCAB_MIN_FREQUENCY)

# module
encoder = Encoder(DATA_SIZE, HIDDEN_SIZE, ENC_DROPOUT)
attention = Attention(HIDDEN_SIZE)
decoder = Decoder(HIDDEN_SIZE, DATA_SIZE, attention, DEC_DROPOUT)
model1 = S2S(encoder, decoder, DATA_SIZE, GENERATION_LEN, device).to(device)

# initialize parameter
model1.apply(init_weights)

# define loss function
PAD_IDX = DATA.vocab.stoi['<pad>']
criterion1 = nn.NLLLoss(ignore_index=PAD_IDX, reduction="sum")

# train
iters(model1, criterion1, LEARNING_RATE, LOAD_FROM_CHECKPOINT, EPOCHS, train_dataset, CLIP, TEACH_RATE,
      CHECKPOINT_TIMES, MODEL_SAVE_PATH, PARA_SAVE_PATH, CHECKPOINT_PATH)

# evaluation
generation_text = eval(model1, test_dataset)


def print_res(gen):
    res = []
    for i in range(TEST_BATCH_SIZE):
        res.append([])
    for iter in gen:
        for j in range(len(iter)):
            word = DATA.vocab.itos[iter[j].cpu().numpy().tolist()]
            res[j].append(word)
    sents = []
    for b in res:
        sentence = ""
        for word in b:
            sentence += word
            sentence += ' '
            if word == '<eos>':
                sents.append(sentence)
                break
        if '<eos>' not in sentence:
            sents.append(sentence)
    for sentence in sents:
        print(sentence)
        print('/n')


print_res(generation_text)
