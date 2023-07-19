# US-Accidents-Severity-Classification

Abstract:
Roads are shared by various means of transportation such as cars, trucks, buses, motorcycles,
pedestrians, and animals, and they play a significant role in the economic and social growth
of many nations. However, every year, a large number of vehicles are involved in collisions
that result in numerous fatalities and injuries. There has been a rise in the number of road
accidents, posing a challenge to governments, individuals, and communities as these
accidents can be deadly and hazardous to society. This paper aims to address this issue by
examining the primary factors that contribute to the increase in the rate of car accidents and
develop a predictive model that accurately identifies accident-prone areas and helps reduce
the frequency and severity of accidents in the USA. The data utilized in this study was
obtained from traffic accidents recorded by the United States Department of Transportation,
law enforcement agencies, and traffic cameras between 2016 and 2021 through multiple data
providers that include various APIs which provide streaming traffic event data and US
Census Demographic data that contains information regarding the population of each state
and county in the United States. The models utilized in this research are the Apriori algorithm
to provide recommendations based on the association rules, the Decision tree classifier with
SMOTE for addressing the class imbalance, and the BERT model to predict the consequences
of car accidents on road traffic, with a particular emphasis on identifying the main factors that
contribute to road accidents. The decision tree classifier despite balanced classes done with
SMOTE gave a low accuracy of 71%. The BERT model for solving this problem showed a
greater accuracy for very fewer data with 85% accuracy in classifying the severity of
accidents. The findings from this study can potentially inform the development of strategies
and interventions aimed at reducing the frequency and severity of car accidents in urban
areas. These could include initiatives such as encouraging the adoption of autonomous
vehicles, improving public transportation infrastructure, and implementing measures to
spread out peak traffic times.

Datasets:
US-Accidents dataset - This large-scale database was created
by gathering, integrating, and supplementing data through a comprehensive process. It
includes information on 2.8 million traffic accidents that occurred in the contiguous United
States. Each accident record contains intrinsic and contextual attributes, such as location,
time, weather, natural language description, points of interest, and time of day. This dataset
on traffic accidents covers 49 states of the United States and is continuously updated since
February 2016. It is collected through multiple data providers that include various APIs
which provide streaming traffic event data. The APIs capture traffic events from different
sources such as traffic cameras, traffic sensors in the road network, and law enforcement
agencies, among others.
It is the collection of car accident data from various sources such as MapQuest and
Bing. The data set was collected from February 2016 to December 2019 and includes
information on traffic events recorded by different entities. The data set contains 47 features,
and details on each feature are given below. Some features include TMC, which is a Traffic
Message Channel code, Severity, which is a number ranging from 1 to 4 indicating the extent
of the impact on traffic and the level of damage or fatalities; and Description, which provides
a natural language description of the accident, and Weather Condition, which describes the
weather at the time of the accident using natural language keywords.
![image](https://github.com/SadakhyaNarnur/US-Accidents-Severity-Classification/assets/111921205/a9029f03-179a-458b-bfec-f11b9560b715)

US Census Demographic dataset - collected from the American
Community Survey (ACS) for the year 2017. It covers all 52 states of America as well as DC
and Puerto Rico. The dataset has two tables for each year and four tables in total. The first
two tables are called census_tract_data which contain data for all census tracts within the US.
The second two tables are called county_data which contain data for all counties or county
equivalents in the US. Although the accidents we are looking at happened over the span of
five years (2016-2021), we do think that the 2017 population data is a decent representation
for all of our analysis since we think that the population would not have changed by a
significant amount. Each table of the US Census Demographic dataset consists of 37 columns
which are identical across the four tables except for the ID column being Census Tract ID for
the two census_tract_data tables and County Census ID for the two county_data tables.
![image](https://github.com/SadakhyaNarnur/US-Accidents-Severity-Classification/assets/111921205/722879b1-33f8-4265-a716-3a6a65d0bc31)

**Apriori algorithm to provide recommendations based on the association rules:**
Apriori Algorithm:
The Apriori algorithm is a classic algorithm used in data mining to find frequent item
sets in a large dataset. It is based on the principle that if an item set is frequent, then all its
subsets must also be frequent.
The algorithm works in two phases. In the first phase, called the "candidate
generation" phase, the algorithm generates a set of candidate itemsets of length k, where k is
the current length of the frequent itemsets. These candidate itemsets are generated by
combining frequent itemsets of length k-1.
In the second phase, called the "candidate pruning" phase, the algorithm scans the
dataset to count the frequency of each candidate itemset generated in the first phase. If an
itemset does not meet the minimum support threshold, it is discarded as a non-frequent
itemset. The frequent itemsets generated in this phase are used to generate candidate itemsets
of length k+1, and the process continues until no more frequent itemsets can be found.
Apriori Algorithm has three parts:
1. Support - Fraction of transactions that contain an itemset.
For example, the support of item I is defined as the number of transactions containing I
divided by the total number of transactions.
Support( I )=
( Number of transactions containing item I ) / ( Total number of transactions )
2. Confidence - Measures how often items in Y appear in transactions that contain X
Confidence is the likelihood that item Y is also bought if item X is bought. It’s calculated as
the number of transactions containing X and Y divided by the number of transactions
containing X.
Confidence( I1 -> I2 ) =
( Number of transactions containing I1 and I2 ) / ( Number of transactions containing I1 )
3. Lift - Measure of association between two items in a frequent itemset. It measures how
much the occurrence of one item in a frequent itemset increases the probability of the other
item in the same frequent itemset.
Lift( I1 -> I2 ) = ( Confidence( I1 -> I2 ) / ( Support(I2) )
Methodology:
The preprocessed data is further processed for making transactions. We start by
replacing the true/false values with string labels that can be self-explanatory. From the
previous modeling, we observed that the absence of bumps, signals, stops, stations, etc. did
not add any importance to the classification hence we remove them from the rules. Further
we use a Transaction encoder which is a class in Python's machine learning library scikitlearn
that is used to convert a list of transactions into a one-hot encoded format suitable for
use in frequent itemset mining algorithms such as Apriori.
The TransactionEncoder class takes as input a list of transactions, where each transaction
is a list of items, and creates a sparse matrix where each row represents a transaction and each
column represents an item. If an item appears in a transaction, the corresponding element in
the matrix is set to 1, otherwise, it is set to 0.
The output of the TransactionEncoder can be fed into an Apriori algorithm to generate
frequent itemsets. Based on the frequent itemsets hence generated we produce the association
rules with ‘lift’ as a metric and sort based on confidence.

**Decision tree classifier for severity classification with SMOTE**
Decision tree Classifier:
A form of machine learning algorithm called a decision tree classifier is used for
supervised learning tasks like classification. Each internal node represents a decision based
on a particular trait, and each leaf node represents a classification label. Together, these nodes
form a tree-like model of decisions and potential outcomes.
Recursively partitioning the input space according to the values of the input characteristics is
how the method constructs the decision tree during training. The objective is to develop
decision rules that correctly forecast the incoming data's class label. Based on a criterion like
information gain or Gini index, the algorithm chooses the optimum feature to partition the
data.
Following the path through the decision tree based on the values of the input
characteristics allows the decision tree to be used to generate predictions on fresh data after it
has been constructed. The algorithm checks the value of each internal node's relevant feature
before moving to the left or right child node depending on whether the value meets a
predetermined requirement. The algorithm outputs the corresponding class label once it
reaches a leaf node. There are many benefits to using decision tree classifiers, including its
usability, interpretability, and capacity for both category and numerical data.
SMOTE :
SMOTE stands for Synthetic Minority Over-sampling Technique, which is a data
augmentation method used in machine learning to address the class imbalance. Class
imbalance occurs when the number of instances in one class is much smaller than the number
of instances in another class, which can lead to poor performance of the model on the
minority class.
SMOTE works by generating synthetic samples for the minority class by interpolating
between existing minority class instances. The basic idea is to randomly select a minority
class instance and then select one or more of its nearest neighbors. Synthetic instances are
then generated by creating linear combinations of the features of the selected instance and its
neighbors, with some random perturbation added to each feature. The result is a set of new
instances that are similar to the existing minority class instances, but not identical.
By increasing the number of minority class instances in this way, SMOTE can help to
balance the distribution of classes in the training data and improve the performance of the
model on the minority class. However, it is important to note that SMOTE should only be
used on training data, and not on the validation or test data, as this can lead to overfitting and
poor generalization performance.
Methodology:
As observed there is a class imbalance with respect to the Severity column with a very
wide disparity between Severity level 2 and rest 1, 3, and 4 as seen in Figure 24.
Figure 24
We initially undersample the severity 2 records to 150000 records and then we
upsample the remaining classes in a range of 15000 to 151145 such that there is a close
balance in the training data. This upsampling is done using SMOTE by specifying a strategy
of desired ratio and fit resampling the data. On this balanced data we split into train and test
splits in 70:30 ratio with a random state of 42.
We use DecisionTreeClassifier from scikit learn library to fit the data. We have used
both entropy and gini index for building the tree with max_depth 8 and random state 1. On
evaluating with the test split an accuracy of 71% is observed for both the criterions.

![image](https://github.com/SadakhyaNarnur/US-Accidents-Severity-Classification/assets/111921205/2b04a5f3-5585-4470-8994-b6726defd175)

Despite handling class balance and using the most widely used classifier it was observed
that the model was unable to learn from the nuances of the features like the presence of
bump, signal, and other categorical features which it considered of less importance. Hence to
propose an approach we experiment with the BERT model next.

**RoBERTa for classifying the severity of the accidents into four categories**
Robustly Optimized Bidirectional Encoder Representations from Transformers
Approach:
Facebook AI Research created RoBERTa, a pre-trained transformer-based neural
network model for tasks involving natural language processing. The popular BERT
(Bidirectional Encoder Representations from Transformers) model's architecture serves as the
foundation for RoBERTa, which is trained on a sizable corpus of text data using a modified
training methodology that incorporates dynamic masking, longer sequences, and other
methods to enhance the model's performance.
The Google-developed pre-trained language model BERT can be adjusted for a range
of natural language processing applications, including text categorization. Similar to BERT,
RoBERTa is adaptable for a range of natural language processing applications, such as text
classification, question resolution, and named entity recognition. RoBERTa is trained on a
particular task during fine-tuning, using a smaller labeled dataset as a starting point rather
than the pre-trained weights.
We must first hone BERT on a particular classification task, such as topic
classification, before we can utilize it for classification. The pre-trained BERT model is then
combined with a task-specific layer, and the entire model is subsequently trained on a labeled
dataset.
BERT receives a sequence of tokens and a label corresponding to the classification
problem as input during training. A probability distribution across the potential labels is the
model's output. The difference between the predicted label and the actual label is measured
by a loss function, and the model is trained to minimize this difference.
Once the model has been trained, it can be applied to new text inputs to make predictions.
Tokenizing and converting the input text into the training data's format comes first. The
output of the task-specific layer is then utilized to create the final prediction once the BERT
model has been applied to the input text.
BERT can capture the context and links between words in a sentence, which can
improve performance on tasks where the meaning of the text is crucial. This is one benefit of
utilizing BERT for text categorization. BERT is widely utilized in both industry and
academics for a variety of text categorization tasks and has attained state-of-the-art results on
several NLP benchmarks.
Methodology:
We start by installing transformers, sentencepiece, contractions and keras
preprocessing. We then import all the required libraries. BERT works by analyzing a long
string of text with all the required details based on which it can be classified. Hence we
preprocess and make a string out of the accidents data with all the features that seem relevant
for classification. The advantage here is it requires no particular data type and accepts the
records in the form of a conversation. Due to high GPU and RAM overhead, we considered a
small fragment of data with balanced classes of 1000 records for each class and an overall
train data of 4000 records. Further some processing specific to BERT training requirements
are done like changing True/False (1,0) to self-explanatory strings say “Bump” and “No
Bump”.
Now we install the Roberta tokenizer that is pretrained and tokenize our text. We then
take tokenized and encoded sentences and attention masks. We now split this data into train
and validation sets. Converting all of our data into torch tensors, the required data type for
our model and creating an iterator of our data with torch DataLoader. This helps save on
memory during training because, unlike a for loop, an iterator the entire dataset does not need
to be loaded into memory. Next we load the model RobertaForSequenceClassification, the
pretrained model will include a single linear classification layer on top for classification. We
next set the custom optimizer Adam and other parameters. Now we train the model and keep
track of the loss. Validation phase comes next where we validate the model and predict the
labels.
Now we prepare the test data and sampled 10 for each class. We preprocess this test
data with the same steps followed for train data preparation. We then tokenize the test text
and predict the classes. Now we find the accuracy and show the classification report. It gave a
better accuracy of 85% for just a small fragment of data compared to the decision tree.

Results:
● The association rules show the relation in terms of support and confidence between
the features.
![image](https://github.com/SadakhyaNarnur/US-Accidents-Severity-Classification/assets/111921205/5f75e641-a6b1-4d39-a3a1-a605e566abf6)

On sorting the rules based on confidence it can be seen that Severity 2 has very little
impact on blockages and delays. Similarly, it was observed that the day and month
had a greater confidence with Severity 4 which helps in finding the fatality of any
accident based on the day and month.
● Decision tree classifier despite balanced classes done with SMOTE gave a low
accuracy of 71% showing that the model couldn't learn how to classify based on the
patterns in the data like bumps, crossing, signals, etc. which are the main features in
road traffic data.
![image](https://github.com/SadakhyaNarnur/US-Accidents-Severity-Classification/assets/111921205/e6a6ba0a-9ee1-4b48-928e-0658b2d7246d)

● Our proposed BERT model for solving this problem showed a greater accuracy for
very fewer data. With the availability of better computing power and GPU resources,
the model can be fine-tuned and better trained with larger data and achieve greater
accuracy. It currently gives 85% accuracy in classifying the severity of accidents.
![image](https://github.com/SadakhyaNarnur/US-Accidents-Severity-Classification/assets/111921205/0f2540f3-8e2e-4cc2-a8da-e642f31f8aab)

Conclusion:
Our proposed approach BERT for solving the severity classification of accidents has
shown great results of 85% accuracy, especially a perfect F1 score in identifying the fatal
accidents that are considered severity 4. This proves that our intuition of trying the
transfer learning of RoBERTa, a generally classification model for Natural Language, can
also work for a specific domain like in this case the accident data.
