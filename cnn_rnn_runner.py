from validation import compute_f1
from preprocessing import *
from DataGenerator import ProcessingSequence
from models import *
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

early_stopping = EarlyStopping(patience = 10) # 조기종료 콜백함수 정의

epochs = 200

trainSentences = readfile("data/train.txt")
validationSentences = readfile("data/valid.txt")
testSentences = readfile("data/test.txt")

# trainSentence : list of [<class 'list'>: [['SOCCER', 'O\n'], ['-', 'O\n'], ['JAPAN', 'B-LOC\n'], ['GET', 'O\n'], ['LUCKY', 'O\n'], ['WIN', 'O\n'], [',', 'O\n'], ['CHINA', 'B-PER\n'], ['IN', 'O\n'], ['SURPRISE', 'O\n'], ['DEFEAT', 'O\n'], ['.', 'O\n']] .....

trainSentences = addCharInformation(trainSentences)
validationSentences = addCharInformation(validationSentences)
testSentences = addCharInformation(testSentences)

labelSet = set()
words = {}

'''
extact label from data
labelset {'I-LOC\n', 'B-PER\n', 'B-MISC\n', 'I-PER\n', 'O\n', 'B-ORG\n', 'B-LOC\n', 'I-ORG\n', 'I-MISC\n'}
'''
for dataset in [trainSentences, validationSentences, testSentences]:
    for sentence in dataset:
        for token,char,label in sentence:
            labelSet.add(label)
            words[token.lower()] = True
'''
words = {'eu': True, 'rejects': True, 'german': True, 'call': True, 'to': True, 'boycott': True, 'british': True, 'lamb': True, ....
'''

# :: Create a mapping for the labels ::
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)

'''
label2Idx = {dict} {'I-LOC\n': 0, 'B-PER\n': 1, 'B-MISC\n': 2, 'I-PER\n': 3, 'O\n': 4, 'B-ORG\n': 5, 'B-LOC\n': 6, 'I-ORG\n': 7, 'I-MISC\n': 8}
labelSet = {set} {'I-LOC\n', 'B-PER\n', 'B-MISC\n', 'I-PER\n', 'O\n', 'B-ORG\n', 'B-LOC\n', 'I-ORG\n', 'I-MISC\n'}
'''

# :: Hard coded case lookup ::
case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7}
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')
''' 
[[1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1.]]
'''


word2Idx={}
wordEmbeddings = []

char2Idx = {"PADDING":0, "UNKNOWN":1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx)
#{'PADDING': 0, 'UNKNOWN': 1, ' ': 2, '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12, 'a': 13, 'b': 14, 'c': 15, 'd': 16, 'e': 17, 'f': 18, 'g': 19, 'h': 20, 'i': 21, 'j': 22, 'k': 23, 'l': 24, 'm': 25, 'n': 26, 'o': 27, 'p': 28, 'q': 29, 'r': 30, 's': 31, 't': 32, 'u': 33, 'v': 34, 'w': 35, 'x': 36, 'y': 37, 'z': 38, 'A': 39, 'B': 40, 'C': 41, 'D': 42, 'E': 43, 'F': 44, 'G': 45, 'H': 46, 'I': 47, 'J': 48, 'K': 49, 'L': 50, 'M': 51, 'N': 52, 'O': 53, 'P': 54, 'Q': 55, 'R': 56, 'S': 57, 'T': 58, 'U': 59, 'V': 60, 'W': 61, 'X': 62, 'Y': 63, 'Z': 64, '.': 65, ',': 66, '-': 67, '_': 68, '(': 69, ')': 70, '[': 71, ']': 72, '{': 73, '}': 74, '!': 75, '?': 76, ':': 77, ';': 78, '#': 79, "'": 80, '"': 81, '/': 82, '\\': 83, '%': 84, '$': 85, '`': 86, '&': 87, '=': 88, '*': 89, '+': 90, '@': 91, '^': 92, '~': 93, '|': 94}

# :: Read in word embeddings ::
wordEmbeddings = embedding_word(path="embeddings/glove.6B.100d.txt",word2Idx=word2Idx,words=words,wordEmbeddings=wordEmbeddings)

'''
#word2vec in wordEmbeddings
[-0.01833607  0.04383408  0.06356397  0.03637314  0.18562967  0.14205369
 -0.01238122 -0.18194667  0.23130247  0.09381868 -0.20482783 -0.01209812
 -0.04383898 -0.0755998   0.24900102 -0.12089226 -0.22637958 -0.22276793
 -0.0953966   0.04842478  0.17808746 -0.0407055  -0.06473728  0.16649523
  0.19523753  0.21066889 -0.17445963 -0.16017444  0.05871271  0.20305255
  0.15933414  0.00312463  0.00703702  0.02098732  0.21424832 -0.18575202
  0.07770112  0.10898197 -0.21717297  0.14165869 -0.07552744  0.10693337
  0.11088641 -0.02070561 -0.04448175 -0.16782276  0.07350525 -0.10086722
  0.09925135  0.19773839  0.10006608  0.07038753 -0.0357064   0.02169009
  0.1285098  -0.07904586  0.24354816 -0.16634841 -0.2167625   0.23438958
 -0.12086422 -0.02937319  0.09371758 -0.16066107 -0.03896686 -0.16680377
 -0.13109609  0.15825993  0.11610326  0.12564714  0.19897688 -0.07700349
  0.09992    -0.05298221 -0.08348135  0.08862719 -0.07069605  0.14400143
 -0.03037521  0.15676496 -0.11295602 -0.10...
{'PADDING_TOKEN': 0, 'UNKNOWN_TOKEN': 1, 'the': 2, ',': 3, '.': 4, 'of': 5, 'to': 6} ....
'''


train_set = padding(createMatrices(trainSentences,word2Idx,  label2Idx, case2Idx,char2Idx,"CNN"))
validataion_set = padding(createMatrices(validationSentences, word2Idx, label2Idx, case2Idx, char2Idx,"CNN"))
test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx,char2Idx,"CNN"))

train_batch,train_batch_len = createBatches(train_set)
validataion_batch, validataion_batch_len = createBatches(validataion_set)
test_batch,test_batch_len = createBatches(test_set)

model = gen_CNN_RNN_model(wordEmbeddings=wordEmbeddings,char2Idx=char2Idx,label2Idx=label2Idx,case2Idx=case2Idx)
plot_model(model, to_file='cnn_rnn.png')
training_generator = ProcessingSequence(train_batch,train_batch_len,"CNN")
validation_generator = ProcessingSequence(validataion_batch, validataion_batch_len,"CNN")
hist = model.fit_generator(generator=training_generator,verbose=1,epochs=epochs, validation_data=validation_generator,callbacks=[early_stopping],workers=10,use_multiprocessing=True)

idx2Label = {v: k for k, v in label2Idx.items()}

#   Performance on test dataset
predLabels, correctLabels = tag_dataset(test_batch,model,"CNN")
pre_test, rec_test, f1_test= compute_f1(predLabels, correctLabels, idx2Label)
print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))


fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

plt.savefig("cnn_rnn_hist.png", dpi=300)
