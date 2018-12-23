from validation import compute_f1
from preprocessing import *
from DataGenerator import ProcessingSequence
from models import *
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

early_stopping = EarlyStopping(patience=10) # 조기종료 콜백함수 정의

epochs = 200

trainSentences = readfile("data/train.txt")
validationSentences = readfile("data/valid.txt")
testSentences = readfile("data/test.txt")

trainSentences = addCharInformation(trainSentences)
validationSentences = addCharInformation(validationSentences)
testSentences = addCharInformation(testSentences)

labelSet = set()
words = {}

for dataset in [trainSentences, validationSentences, testSentences]:
    for sentence in dataset:
        for token,char,label in sentence:
            labelSet.add(label)
            words[token.lower()] = True

# :: Create a mapping for the labels ::
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)

# :: Hard coded case lookup ::
case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7}
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')

word2Idx={}
wordEmbeddings = []

char2Idx = {"PADDING":0, "UNKNOWN":1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx)

# :: Read in word embeddings ::
wordEmbeddings = embedding_word(path="embeddings/glove.6B.100d.txt",word2Idx=word2Idx,words=words,wordEmbeddings=wordEmbeddings)

train_set = padding(createMatrices(trainSentences,word2Idx,  label2Idx, case2Idx,char2Idx,"RNN"))
validataion_set = padding(createMatrices(validationSentences, word2Idx, label2Idx, case2Idx, char2Idx,"RNN"))
test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx,char2Idx,"RNN"))

train_batch,train_batch_len = createBatches(train_set)
validataion_batch, validataion_batch_len = createBatches(validataion_set)
test_batch,test_batch_len = createBatches(test_set)
training_generator = ProcessingSequence(train_batch,train_batch_len)
validation_generator = ProcessingSequence(validataion_batch, validataion_batch_len)


model = gen_RNN_RNN_model(wordEmbeddings=wordEmbeddings,caseEmbeddings=caseEmbeddings,label2Idx=label2Idx)
plot_model(model, to_file='rnn_rnn.png')
hist = model.fit_generator(generator=training_generator,verbose=1,epochs=epochs, validation_data=validation_generator,callbacks=[early_stopping],workers=10,use_multiprocessing=True)



idx2Label = {v: k for k, v in label2Idx.items()}

#   Performance on test dataset
predLabels, correctLabels = tag_dataset(test_batch,model)
pre_test, rec_test, f1_test= compute_f1(predLabels, correctLabels, idx2Label)
print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))


fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

plt.savefig("rnn_rnn_hist.png", dpi=300)