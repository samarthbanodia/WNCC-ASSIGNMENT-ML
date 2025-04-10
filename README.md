For the ML model to understand the text we need to preprocess the data and represent in format that the model can understand. Preprocessing is mainly just cleaning the data and making it simpler for further steps, this reduces the computional power required and makes the model slightly faster.

#### My approach - after reading the assignment i made my way to youtube to find out what all the preprocessing , data rep methods do and how to write them in code. I decided to work with tensorflow as i have worked with it before , I decided to build a NN model for the assignment as its more suitbale for Text classification on a relatively smaller training set. I knew what vector embedding and tokenizations were as i had learnt someof the stuff before watching 3b1b videos . I knew i had to use regex to clean the data as i used it in my WIDS project as well. After that i generated the sample csv files from chatgpt and for the eval metric part i found that scikit learn has modules for precision , recall , f1 which can  be used directly .  


Exploring the Preprocessing methods,
 - Tokenization - splits the sentence into words and seperate tokens. eg: I love wncc <3 --> 'I' , 'love', 'wncc' , '<3'.
 - Stopword Removal - removes useless words like 'and', 'or', 'the' which dont affect the meaning that much.
 - Stemming - reduce words to their smallest version eg: running to run  ,studying to study
 - Lemmatization - reduce words to their meaning base eg: sprinting - run


Here i will be using Tokenization with help of Tokenizer form tensorflow.keras.preprocessing.text. 
https://www.youtube.com/watch?v=U3ZjAbf-0J4
https://www.youtube.com/watch?v=9ieVC_ABDNQ - padding for uniformity

but before tokenization i will be making the text lowercase , removing the punctuation and removing stopwords.

After searching on yt i found a library nltk which have a pre made stopwords list. https://www.youtube.com/watch?v=hhjn4HVEdy0

to remove the punctuation i remembered to use regex as i used the same in my WIDS project (Advanced NLP for beginner). r'[^a-zA-Z0-9\s]' means characters except for a-z A-Z and 0-9 and whitespaces.


Coming to Text Representation ,

- Tf-IDF - based on frequency of term - not very useful with smaller training set like in our case.
- Word2vec - converts each into a high dimensional vector

Here im using Embedding layers from tensorflow
https://www.youtube.com/watch?v=Fuw0wv3X-0o
https://www.youtube.com/watch?v=sZGuyTLjsco

`model = tf.keras.Sequential([

    tf.keras.layers.Embedding(input_dim=1000, output_dim=16),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dense(6, activation='softmax'),

])`

5 layers model with softmax in output layer .
I chose this method as other method which uses semantic analysis arent of use what these are queries.


#### Model i used,

I used a neural network model made in tensorflow , it contains an initial Embedding layers which converts words to dense vectors. then comes 2 hidden layers with reLu activation functions and then output layers with softmax Activation to map output to (0,1) to categorize into the 6 classes.
 NN is easy to build as well as easy to train .

The reason i didnt use other models like logistic regression , naive bayes , random forest etc is logistic reg is more usefull for binary classes rather than multiple classes and also less usefull for a smaller set . Random forset - not fit for text sequences.

Challenges i foresee -  finding  the right number of hidden layer , activation functions etc for more accurate model can be difficult. Model not being accurate enough.

#### Eval Metric to use , 

- Accuracy - gives the overall view of performance of the model in predicting correct labels.
- Precision - measures how many times the predicion was correct
- Recall - ability to predict true positives 
- F1- score harmonic mean of precision and recall

#### If a model performs poorly we can,
- Hyperparameter tuning- we can tweak the number of epochs , Number of hidden layers , number of neurons in that layer ,  try changing the activation function.
- Collecting more data - increasing you training set can often imporve the accuracy
- If nothing is working can try different model other than NN like logistic reg , random forest etc
- Data cleaning - we can clean data even more for more accurate training

#### Performance of my model::

![[Pasted image 20250410180540.png]]
ignore the warning .

I tried tuning the hyperparameter , increasing the epochs  , adding more layers with different numner of neurons but the accuracy didnt seem to go beyond 0.2 .

