# About Project :
Sentiment analysis project using BERT model and pytorch . Here used Google PlaystoreAPP Review Dataset and from the dataset  used review  column to predict the sentiment of the review.

1. Used BERT-Base-Cased version of BERT Tokenizer to tokenize ,build the sequence of intergers of these tokens and others pre-processing steps which is required by
BERT model to feed input for training purposes.

2. Used  BERT-Base-Cased version of BERT pretrained model to make context ( pooled output) from the  tokenized reviews.

3. Used a Dense layers to predict the predictions score of each classes for each reviews.

4. Build a model using  BERT and Dense layer and train the model using DataModeule and Dataloader from pytorch.

5. Used the tranied model to make prediction from new data input (reviews).



