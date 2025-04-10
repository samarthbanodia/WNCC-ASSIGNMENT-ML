import pandas as pd
import re #regex lib
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences as pad #for uniformity
import tensorflow as tf

stop_words = set(stopwords.words('english'))
predicted_cat = ['0','0','0','0','0']

categories = ['DSA', 'ML', 'Web Dev', 'Comp Programming', 'Open Source', 'General']


def preprocessing_text(text):
    text = text.lower()       #lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  #remove punctuation
    words = text.split()
    words = [w for w in words if w not in stop_words]   #remove stopword
    return ' '.join(words)

df = pd.read_csv("queries.csv")
df['clean_text'] = df['text'].apply(preprocessing_text) #apply method for a function in pandas!!

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
padded = pad(sequences, padding='post')

labels = df['label'].values

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')  # maps to (0,1) for 6 dimensional output (6 classes)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #spare_categorical loss function because we have sparse classes not 0,1,1,1 ,0 types (binary)

# Step 5: Train
model.fit(padded, labels, epochs=20) #20 passes

def predict(text):
    cleaned_text = preprocessing_text(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad(seq, maxlen=padded.shape[1], padding='post')
    pred = model.predict(padded_sequence)
    return tf.argmax(pred[0]).numpy()  #argmax is to findout the highest activation output neuron for the class.

test_data = pd.read_csv("test_queries.csv")
for i in range(len(test_data)):
    query = test_data['text'][i]
    predicted = predict(query)
    sum=0
    print(f"query: {query} ; Predicted cat: {categories[predicted]} ; Actual cat: {categories[test_data['label'][i]]}")
    predicted_cat[i] = predicted
    if predicted == test_data['label'][i] :
        sum=sum+1
    
    
print(f"accuracy:  {sum/5}")    
    
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(test_data['label'], predicted_cat, average='weighted')
recall = recall_score(test_data['label'], predicted_cat, average='weighted')
f1 = f1_score(test_data['label'], predicted_cat, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
