#%%Package loading

#bert model
from pandas.core.frame import DataFrame
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F 
#basic data
import numpy as np
import pandas as pd
#preprocessing
from collections import defaultdict
from textwrap import wrap
#plotting
import seaborn as sns
from pylab import rcParams #??
import matplotlib.pyplot as plt
#modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
#%%
#text topics
import re

from transformers.utils.dummy_pt_objects import BERT_PRETRAINED_MODEL_ARCHIVE_LIST
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
#%% Config
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Data exploration
df = pd.read_csv('https://raw.githubusercontent.com/kylin233chen/crypto_market_text_analysis/main/crypto_news.csv')
inter = pd.to_datetime(df['date'], errors='coerce',utc=True)
df['date'] = inter.dt.strftime('%Y-%m-%d')
df['sentiment'] = df['sentiment'].astype('category')

#%%
##--------------------- top topics -------------------##
top_topics = df['topics'].value_counts().head(10)
#there's 11809 null topic
#topic allocation? --lda
news_for_lda = df[['date','title','text','topics','sentiment','source_name', 'tickers']]
text_conten_only =pd.DataFrame(news_for_lda['title']+news_for_lda['text'])
text_conten_only.columns = ['text_content']
#%%
all_content = pd.DataFrame( text_conten_only['text_content'].map(lambda x:x.lower()).map(lambda x: re.sub('[^a-z\\s]','',x)) )
all_content.columns = ['text']

#%%
##------------------------BERT-----------------------##
ax = sns.countplot(df.sentiment)
def senti_score(sentiment):
    if sentiment == 'Positive':
        return 0
    elif sentiment == 'Neutral':
        return 1
    else:
        return 2

df_bert = pd.concat([text_conten_only['text_content'],df['sentiment']],axis=1)
df_bert['sentiment_target'] = df_bert.sentiment.apply(senti_score)
#%%

# let's encoding
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
encoding = tokenizer.encode_plus('sample_txt',
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  max_length=305, # we should discuss on that
  padding='max_length',
  return_attention_mask=True,
  return_tensors='pt',  # Return PyTorch tensors
  truncation=True
)
encoding.keys()
# play around tokens
# tokenizer.convert_ids_to_tokens(encoding.input_ids[0])


# %%
# choose the lenth to represent
# make sure all the text could be incorporated (longest)
len_for_tokens = []
for i in df_bert['text_content']:
    token_len = len(tokenizer.encode(i))
    len_for_tokens.append(token_len)

#sns.displot(len_for_tokens)
max(len_for_tokens) #305
torch.tensor(2, dtype=torch.long)
# %% encoding func
class GPReviewDataset(Dataset):
  def __init__(self, text_content , sentiment_target, tokenizer, max_len):
    self.text = text_content
    self.targets = sentiment_target
    self.tokenizer = tokenizer
    self.max_len = max_len
  def __len__(self):
    return len(self.text)
   
  def __getitem__(self, item):
    text_con = str(self.text[item])
    sen_target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      text_con,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'review_text': text_con,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(sen_target, dtype=torch.long)
    }

#%% set splitting
df_train, df_test = train_test_split(
  df_bert,
  test_size=0.1,
  random_state=RANDOM_SEED
)
df_val, df_test = train_test_split(
  df_test,
  test_size=0.5,
  random_state=RANDOM_SEED
)
df_train.shape, df_val.shape, df_test.shape
#%%
def create_data_loader(df, tokenizer, Max_len, batch_size):
  ds = GPReviewDataset(
    df.text_content.to_numpy(),
    df.sentiment_target.to_numpy(),
    tokenizer,
    Max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0 #Windows cannot run this DataLoader in 'num_workers' more than 0. 
  )

#%% ------------data set ready-------------------------##
BATCH_SIZE = 6
MAX_LEN = 305
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

#%%
data=next(iter(train_data_loader))
data.keys()
data['input_ids'].shape
#%%load bert model
bert_model = BertModel.from_pretrained('bert-base-cased')

#%%
class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = bert_model
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=False

    )
    output = self.drop(pooled_output)
    return self.out(output)
    
#%%
model = SentimentClassifier(3)
model = model.to(device)
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)
#%%
F.softmax(model(input_ids, attention_mask), dim=1)

#%% training epochs & loss func
EPOCHS = 15
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

#%% epoch & evaluation
def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
  model = model.train()
  losses = []
  correct_predictions = 0
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)

#%% ----start training----
%%time
history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)
  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
  )
  print(f'Train loss {train_loss} accuracy {train_acc}')
  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
    device,
    len(df_val)
  )
  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)
  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'best_model_state.bin')
    best_accuracy = val_acc
