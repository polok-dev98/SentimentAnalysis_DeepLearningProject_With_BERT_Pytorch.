# Install the hugging face transformer and pytorch lighting
#!pip install  transformers

#importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import transformers
from transformers import BertTokenizer,BertModel,AdamW,get_linear_schedule_with_warmup
import tqdm.auto as tqdm
import torch.nn.functional as F
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn,optim
from textwrap import wrap
from torch.utils.data import Dataset, DataLoader

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 8, 4
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
import warnings
warnings.filterwarnings('ignore')

#Load the dataset
df=pd.read_csv("reviews.csv")

#check for missing values
df.info()
#Great, no missing values in the score and review texts!

#check for imblanced classes
sns.countplot(df.score)
plt.xlabel('review score') #That's imbalanced, but it's okay. We're going to convert the dataset into negative, neutral and positive sentiment 😉😊😊

#grouping the labels
def to_sentiment(rating):    
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else: 
        return 2

df['sentiment'] = df.score.apply(to_sentiment)

#now check wheather the classes is balanced or not
class_names = ['negative', 'neutral', 'positive']

ax = sns.countplot(df.sentiment)
plt.xlabel('review sentiment')
ax.set_xticklabels(class_names)

#--------------------------Data Preprocessing--------------------------------------

PRE_TRAINED_MODEL_NAME = 'bert-base-cased' #small version of bert
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#Choosing Sequence Length
#BERT works with fixed-length sequences. We'll use a simple strategy to choose the max length. Let's store the token length of each review.
token_lens = []

for txt in df.content:
    tokens = tokenizer.encode(txt,max_length=512) #each row contain fixed length of 512
    token_lens.append(len(tokens))
    
sns.distplot(token_lens)
plt.xlim([0, 256]);
plt.xlabel('Token count')    

#---------------------------Create dataset------------------------------------

MAX_LEN = 160

class GPReviewDataset(Dataset):   
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
  
    def __len__(self):
        return len(self.reviews)
  
    def __getitem__(self, index):
        review = str(self.reviews[index])
        target = self.targets[index]

        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

        return {
          'review_text': review,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
                }

#split the data 
df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

#--------------------------Create Data Loader-------------------------------

def create_data_loader(df, tokenizer, max_len, batch_size):
    #create a instance of the GPReview class.
    ds = GPReviewDataset(
        reviews=df.content.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len)
    
    return DataLoader(
        ds,
        batch_size=batch_size)

BATCH_SIZE = 16

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

#--------------------------------Building the model--------------------------------

#initialize the model
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,return_dict=False)

class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,return_dict=False)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
    def forward(self, input_ids, attention_mask):
        
        _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,return_dict=False)
        output = self.drop(pooled_output)
        return self.out(output)

model = SentimentClassifier(len(class_names))
#model = model.to(device)

#-----------------------------------Training----------------------------------

EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss()

# Traning function
def train_epoch(model, 
                data_loader, 
                loss_fn, 
                optimizer,  
                scheduler, 
                n_examples):
    
    model = model.train() # model becoming in the training mode.

    losses = []
    correct_predictions = 0
  
    for batch in data_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        #avoiding the exploiding gradiant
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


# Evaluation function
def eval_model(model, data_loader, loss_fn, n_examples):
    
    model = model.eval() # model becoming in the evaluation mode.(no dropout,batch norm apply)

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            targets = batch["targets"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)
    

# starting the training process.
from collections import defaultdict
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
        scheduler, 
        len(df_train))

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn, 
        len(df_val))

    print(f'Val  loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc


# visualize the training and validation accuracy 

plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);        

#load the save models

model = SentimentClassifier(len(class_names))
model.load_state_dict(torch.load('best_model_state.bin'))

#------------------------------------Evaluation------------------------------
# how good is our model on predicting sentiment? Let's start by calculating the accuracy on the test data.
test_acc, _ = eval_model(
  model, # use the trained model.
  test_data_loader,
  loss_fn,
  len(df_test)
)

print(test_acc.item())

#---------------------generate the predictions------------------------------------

def get_predictions(model, data_loader):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = [] #targets

  with torch.no_grad():
      for d in data_loader:
          texts = d["review_text"]
          input_ids = d["input_ids"]
          attention_mask = d["attention_mask"]
          targets = d["targets"]
          
          #outputs = logits(each class prediction score)
          outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
          )
          _, preds = torch.max(outputs, dim=1)

          probs = F.softmax(outputs, dim=1)

          review_texts.extend(texts)
          predictions.extend(preds)
          prediction_probs.extend(probs)
          real_values.extend(targets)
  
  #convert a sequence of tensors(lists) (two or more tensors) into a new dimension(single tensor). 
  # joins the tensors with the same dimensions and shape.
  predictions = torch.stack(predictions)
  prediction_probs = torch.stack(prediction_probs)
  real_values = torch.stack(real_values)
  
  return review_texts, predictions, prediction_probs, real_values

# call the function
y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader)     
      
# check the classification report 
print(classification_report(y_test, y_pred, target_names=class_names))

# check the confusion matrix
def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment');

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

#----------------------------show the prediction-----------------------------------

# set sample index for testing (from test dataset)
idx = 2
review_text = y_review_texts[idx]
true_sentiment = y_test[idx]
pred_df = pd.DataFrame({
  'class_names': class_names,
  'values': y_pred_probs[idx]
})

print("\n".join(wrap(review_text)))
print()
print(f'True sentiment: {class_names[true_sentiment]}')

# visualize the result
sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
plt.ylabel('sentiment')
plt.xlabel('probability')
plt.xlim([0, 1]);

#-------------------------------Predicting on Raw Text------------------------

def predictins(review_text):
    
    encoded_review = tokenizer.encode_plus(
      review_text,
      max_length=MAX_LEN,
      add_special_tokens=True,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt' )

    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    print(f'Review text: {review_text}')
    print(f'Sentiment  : {class_names[prediction]}')

# set your review for predicting the sentiment.
review_text = "I love completing my todos! Best app ever!!!"

# call for the review.
predictins(review_text)

#-------------------------------------END----------------------------------------------


