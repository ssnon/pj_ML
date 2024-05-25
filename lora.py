from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
import evaluate
import torch
import numpy as np
import os
from datasets import concatenate_datasets,DatasetDict
from itertools import islice
import json
import gzip
import torch
import torch.nn as nn

import math
import warnings
from dataclasses import dataclass
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
import torch.optim as optim

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_checkpoint = "bert-base-uncased"
lr = 1e-4
batch_size = 16
num_epochs = 1000
load_model=False
#output_dir = 'kwwww/test_16_2000'
r_=1
lora_alpha=16
num_layer=12
device = torch.device("cuda")
main_path = 'output/'
load_path = 'output-save/model_weights_epoch4.pth'

bionlp = load_dataset("imdb")

bionlp_test1 = bionlp['test'].shard(num_shards=5, index=0)#5000
bionlp_test2 = concatenate_datasets([bionlp_test1, bionlp['test'].shard(num_shards=5, index=1)])#10000
bionlp_test3 = concatenate_datasets([bionlp_test2, bionlp['test'].shard(num_shards=5, index=2)])#15000

bionlp_test_1 = bionlp['test'].shard(num_shards=5, index=3)
bionlp_test_2 = concatenate_datasets([bionlp_test_1, bionlp['test'].shard(num_shards=5, index=4)])#10000

bionlp['train'] = concatenate_datasets([bionlp['train'], bionlp_test3])#2000
#bionlp['test'] = bionlp_test_2


imdb = bionlp

imdb_split_100 = imdb['train'].shard(num_shards=400, index=0) #100
imdb_split_200 = imdb['train'].shard(num_shards=200, index=0) #200
imdb_split_500 = imdb['train'].shard(num_shards=80, index=0) #500
imdb_split_1000 = imdb['train'].shard(num_shards=40, index=0) #1000
imdb_split_2000 = imdb['train'].shard(num_shards=20, index=0) #100
imdb_split_4000 = imdb['train'].shard(num_shards=10, index=0) #100
imdb_split_4500 = imdb['train'].shard(num_shards=9, index=0) #100
imdb_split_5000 = imdb['train'].shard(num_shards=8, index=0) #10
imdb_split_5800 = imdb['train'].shard(num_shards=7, index=0) #100
imdb_split_6500 = imdb['train'].shard(num_shards=6, index=0) #100
imdb_split_8000 = imdb['train'].shard(num_shards=5, index=0) #100
imdb_split_10000 = imdb['train'].shard(num_shards=4, index=0) #100
imdb_split_20000 = imdb['train'].shard(num_shards=2, index=0) #100

imdb_split_test = imdb['test'].shard(num_shards=4, index=0)

#imdb['train'] = imdb_split_20000
#imdb['test'] = imdb_split_test

bionlp = imdb
print(bionlp)

split_size = bionlp['train'].shape[0]
output_dir =  f'kwwww/{model_checkpoint}_{r_}_{split_size}'
seqeval = evaluate.load("seqeval")
accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    return {"f1": f1, "accuracy": acc}
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
    
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
    
tokenized_bionlp = bionlp.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

'''
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
)
'''
original_model_name = 'bert-base-uncased'
original_config = BertConfig.from_pretrained(original_model_name)
custom_config = BertConfig(
    vocab_size=original_config.vocab_size,
    hidden_size=64,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=original_config.intermediate_size,
    max_position_embeddings=original_config.max_position_embeddings,
    type_vocab_size=original_config.type_vocab_size,
    initializer_range=original_config.initializer_range,
    layer_norm_eps=original_config.layer_norm_eps,
    hidden_dropout_prob=original_config.hidden_dropout_prob,
    attention_probs_dropout_prob=original_config.attention_probs_dropout_prob,
)
model = BertForSequenceClassification(custom_config)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

print(":====> reinit")
model.apply(init_weights)

#model.load_state_dict(torch.load(load_path))

class CustumCallback:
    def __init__(self, model):
        self.model = model
    def on_init_end(self, args, state, control, **kwargs):
        pass
    def on_train_begin(self, args, state, control, **kwargs):
        pass
    def on_epoch_begin(self, args, state, control, **kwargs):
        pass
    def on_step_begin(self, args, state, control, **kwargs):
        pass
    def on_step_end(self, args, state, control, **kwargs):
        pass
        
    def on_epoch_end(self, args, state, control, **kwargs):
        output_dir_1 = os.path.join(main_path, "model_weights_epoch4.pth")
        torch.save(model.state_dict(), output_dir_1)
            
    def on_prediction_step(self, args, state, control, **kwargs):
        pass
    def on_log(self, args, state, control, **kwargs):
        pass
    def on_evaluate(self, args, state, control, **kwargs):
        pass
    def on_save(self, args, state, control, **kwargs):
        pass
    def on_train_end(self, args, state, control, **kwargs):
        pass

training_args = TrainingArguments(
    output_dir=main_path,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    load_best_model_at_end=True
)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_bionlp["train"],
    eval_dataset=tokenized_bionlp["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
    callbacks=[CustumCallback(model)],
)

trainer.train()

output_dir_1 = os.path.join(main_path, "model_weights_epoch4.pth")
torch.save(model.state_dict(), output_dir_1)
