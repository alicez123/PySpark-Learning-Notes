
# MLlib
## Feature Extraction
the process of transforming raw data into numerical features that can be used as input for machine learning algorithms. MLlib provides several feature extraction techniques, including:
-   Tokenization: Splitting text into individual words or tokens.
-   Stopword removal: Removing common, uninformative words from text data.
-   TF-IDF: Term frequency-inverse document frequency, a measure of word importance in a document relative to a collection of documents.
-   Word2Vec: A technique for representing words as dense vectors, capturing their semantic meaning.

Example for Tokenization and TF-IDF:
```python
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("SparkFeature").getOrCreate()

# Sample DataFrame with text data
data = spark.createDataFrame([
    (0, "Apache Spark"),
    (1, "Java, Scala, Python, and R"),
    (2, "machine learning")
], ["id", "text"])

# Tokenize the text
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
tokens = tokenizer.transform(data)
tokens.show()
    #+---+--------------------+--------------------+
    #| id|                text|              tokens|
    #+---+--------------------+--------------------+
    #|  0|        Apache Spark|     [apache, spark]|
    #|  1|Java, Scala, Pyth...|[java,, scala,, p...|
    #|  2|    machine learning| [machine, learning]|
    #+---+--------------------+--------------------+
    
# Compute term frequency
hashing_tf = HashingTF(inputCol="tokens", outputCol="rawFeatures")
tf = hashing_tf.transform(tokens)
tf.select('rawFeatures').show(truncate=False)
    #+------------------------------------------------------------------+
    #|rawFeatures                                                       |
    #+------------------------------------------------------------------+
    #|(262144,[68303,173558],[1.0,1.0])                                 |
    #|(262144,[73758,152091,205413,212482,219915],[1.0,1.0,1.0,1.0,1.0])|
    #|(262144,[9144,163984],[1.0,1.0])                                  |
    #+------------------------------------------------------------------+
    
# Compute inverse document frequency and TF-IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(tf)
tf_idf = idf_model.transform(tf)
tf_idf.select('features').show(truncate=False)
    #+---------------------------------------------------------------------------------------------------------------------------------------------+
    #|features                                                                                                                                     |
    #+---------------------------------------------------------------------------------------------------------------------------------------------+
    #|(262144,[68303,173558],[0.6931471805599453,0.6931471805599453])                                                                              |
    #|(262144,[73758,152091,205413,212482,219915],[0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453])|
    #|(262144,[9144,163984],[0.6931471805599453,0.6931471805599453])                                                                               |
    #+---------------------------------------------------------------------------------------------------------------------------------------------+

```
**Note**: there can be **hash collisions** (multiple words mapping to the same hash value), which might result in some loss of information. However, in practice, this technique usually works well and provides a more memory-efficient way of representing term frequencies in large-scale text data.

## Basic statistical
-   Summary statistics: Compute the mean, variance, and other summary statistics for a dataset.
-   Correlations: Measure the linear relationship between two variables.
-   Stratified sampling: Divide a dataset into multiple subsets based on the values of a specific column.

```python
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Summarizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("SparkStatistics").getOrCreate()

data = [(Vectors.dense([0.0, 1.0, 0.5]),),
        (Vectors.dense([1.0, 2.0, 1.5]),),
        (Vectors.dense([2.0, 3.0, 2.5]),)]

data_frame = spark.createDataFrame(data, ["features"])

# Compute summary statistics
summarizer = Summarizer.metrics("mean", "variance")
summary = data_frame.select(summarizer.summary(col("features")).alias("summary")).select("summary.*")

mean = summary.select("mean").collect()[0]["mean"]
variance = summary.select("variance").collect()[0]["variance"]

print("Mean:", mean) # Mean: [1.0,2.0,1.5]
print("Variance:", variance) # Variance: [1.0,1.0,1.0]
 
#spark.stop()

```

## Regression
-   Linear regression
-   Ridge regression
-   Lasso regression

```python
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("LinearRegression").getOrCreate()

data = [(0.0, Vectors.dense([1.0, 2.0, 3.0])),
        (1.0, Vectors.dense([4.0, 5.0, 6.0])),
        (2.0, Vectors.dense([7.0, 8.0, 9.0]))]

data_frame = spark.createDataFrame(data, ["label", "features"])

# Define the linear regression model
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lr_model = lr.fit(data_frame)

# Print the coefficients and intercept for linear regression
print("Coefficients:", lr_model.coefficients)
print("Intercept:", lr_model.intercept)

# Summarize the model over the training set and print some metrics
training_summary = lr_model.summary
print("RMSE:", training_summary.rootMeanSquaredError)
print("R2:", training_summary.r2)

spark.stop()

```

## Classification
-   Logistic regression
-   Naive Bayes
-   Support Vector Machines

```python
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("NaiveBayesIris").getOrCreate()

# Load Iris dataset
data = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(r"C:\Users\username\Anaconda3\Lib\site-packages\bokeh\sampledata\_data\iris.csv")

# Split dataset into training and test sets
train_data, test_data = data.randomSplit([0.7, 0.3])

# Preprocessing
indexer = StringIndexer(inputCol="species", outputCol="label")
vector_assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")

# Initialize Naive Bayes model
naive_bayes = NaiveBayes()

# Create pipeline
pipeline = Pipeline(stages=[indexer, vector_assembler, naive_bayes])

# Train model
model = pipeline.fit(train_data)

# Make predictions on test data
predictions = model.transform(test_data)

# Evaluate model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy)) # Test set accuracy = 0.9583333333333334

# Stop Spark session
spark.stop()
```


## Decision Trees
Decision Trees can be used for both classification and regression tasks. They are particularly useful for handling non-linear relationships and missing values.

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

# Load the data
data = spark.read.format("libsvm").load(r"C:\spark\data\mllib\sample_libsvm_data.txt")

# Index labels, adding metadata to the label column.
label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(training_data, test_data) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[label_indexer, feature_indexer, dt])

# Train model. This also runs the indexers.
model = pipeline.fit(training_data)

# Make predictions
predictions = model.transform(test_data)

# Select example rows to display
predictions.select("prediction", "indexedLabel", "features").show(5)
    #+----------+------------+--------------------+
    #|prediction|indexedLabel|            features|
    #+----------+------------+--------------------+
    #|       1.0|         1.0|(692,[100,101,102...|
    #|       1.0|         1.0|(692,[121,122,123...|
    #|       1.0|         1.0|(692,[124,125,126...|
    #|       1.0|         1.0|(692,[124,125,126...|
    #|       1.0|         1.0|(692,[124,125,126...|
    #+----------+------------+--------------------+

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy)) # Test Error = 0.0625

tree_model = model.stages[2]
print(tree_model)  # summary only
    #DecisionTreeClassificationModel: uid=DecisionTreeClassifier_7f0ae0b9893c, depth=2, numNodes=5, numClasses=2, numFeatures=692

spark.stop()
```

## Recommendation Systems using Alternating Least Squares (ALS)
Collaborative filtering is commonly used for recommendation systems, and ALS is one of the most popular algorithms for collaborative filtering. Here's an example using ALS:

```python
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

lines = spark.read.text(r"C:\spark\data\mllib\als\sample_movielens_ratings.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
ratings_rdd = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratings_rdd)

(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse)) # Root-mean-square error = 1.6229207154131078

# Generate top 10 movie recommendations for each user
user_recs = model.recommendForAllUsers(10)
user_recs.show()
    #+------+--------------------+
    #|userId|     recommendations|
    #+------+--------------------+
    #|    20|[{22, 4.5080743},...|
    #|    10|[{55, 4.0802617},...|
    #|     0|[{9, 3.6954806}, ...|
    #|     1|[{62, 3.9052143},...|
    #|    21|[{53, 4.979208}, ...|
    #|    11|[{18, 5.151548}, ...|
    #|    12|[{55, 6.4718647},...|
    #|    22|[{75, 5.0470934},...|
    #|     2|[{93, 5.162949}, ...|
    #|    13|[{96, 3.9532413},...|
    #|     3|[{32, 4.5733943},...|
    #|    23|[{55, 5.1912537},...|
    #|     4|[{62, 3.9077175},...|
    #|    24|[{29, 5.8899326},...|
    #|    14|[{29, 5.0263796},...|
    #|     5|[{46, 5.665154}, ...|
    #|    15|[{46, 4.8386497},...|
    #|    25|[{25, 4.1402774},...|
    #|    26|[{90, 6.453432}, ...|
    #|     6|[{25, 5.1295414},...|
    #+------+--------------------+
    #only showing top 20 rows

spark.stop()
```

# RDDs
	Resilient Distributed Datasets_
### Initialization
see https://spark.apache.org/docs/latest/configuration.html for all parameters
```python
from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("PySpark RDD Cheatsheet")
sc = SparkContext(conf = conf)

# or
conf = SparkConf() \
    .setAppName("My Spark App") \
    .setMaster("local") \
    .set("spark.executor.memory", "1g") \
    .set("spark.driver.memory", "512m") \
    .set("spark.executor.cores", "2") \
    .set("spark.driver.cores", "1") \
    .set("spark.default.parallelism", "4") \
    .set("spark.driver.maxResultSize", "1g")

sc = SparkContext(conf=conf)
```

### When you finish, stop current sc
	Cannot run multiple SparkContexts at once, so you need to stop the current one in order to start next sc
```python
sc.stop()
```

### Creating RDDs:
```python
# Create an RDD from a list
rdd = sc.parallelize([1, 2, 3, 4, 5])
print(rdd.collect()) # Output: [1, 2, 3, 4, 5]

# Create an RDD from a text file
rdd = sc.textFile("data.txt")
```

### Basic RDD Operations
```python
# Map operation
new_rdd = rdd.map(lambda x: x * 2)
print(new_rdd.collect()) # Output: [2, 4, 6, 8, 10]

# Filter operation
filtered_rdd = rdd.filter(lambda x: x > 2)
print(filtered_rdd.collect()) # Output: [3, 4, 5]

# FlatMap operation
flat_rdd = rdd.flatMap(lambda x: (x, x * 2))
print(flat_rdd.collect()) # Output: [1, 2, 2, 4, 3, 6, 4, 8, 5, 10]

# Map vs flatmap example
	# Map: same number of elements as the original RDD.
	# Flatmap: flattened into a single RDD containing all the values returned by the iterator
sentences_rdd = sc.parallelize(['Hello world', 'How are you'])

words_rdd = sentences_rdd.map(lambda sentence: sentence.split(' '))
print(words_rdd.collect()) # Output: [['Hello', 'world'], ['How', 'are', 'you']]

words_rdd_flat = sentences_rdd.flatMap(lambda sentence: sentence.split(' '))
print(words_rdd_flat.collect()) # Output: ['Hello', 'world', 'How', 'are', 'you']

# Union operation
rdd1 = sc.parallelize([6, 7, 8]) 
rdd2 = sc.parallelize([9, 10, 11])
union_rdd = rdd1.union(rdd2)
print(union_rdd.collect()) # Output: [6, 7, 8, 9, 10, 11]

# Distinct
products_rdd = sc.parallelize([('Apple', 'Fruit'), ('Banana', 'Fruit'), ('Carrot', 'Vegetable')])
distinct_categories_rdd = products_rdd.map(lambda x: x[1]).distinct()
print(distinct_categories_rdd.collect())  # Output: ['Fruit', 'Vegetable']

# length of RDD, number of elements in RDD
rdd.count()
```

### RDD Actions:
```python
# Collect operation
data = rdd.collect()
print(data) # Output: [1, 2, 3, 4, 5]

# Count operation
num_elements = rdd.count()
print(num_elements) # Output: 5

# First operation
first_element = rdd.first()
print(first_element) # Output: 1

# Take operation
first_n_elements = rdd.take(3)
print(first_n_elements) # Output: [1, 2, 3]

# Reduce operation
sum_of_elements = rdd.reduce(lambda a, b: a + b)
print(sum_of_elements) # Output: 15
```

### RDD Transformations:
	Note: when using key related functions such as groupByKey,reduceByKey, the first element is automatically the key, and its value start from the second element.
	However, in map related functions, indexing starts from the first element.
	
	e.g. ("A", "B", "C"), 
	in lambda funtion,
	key related functions:  x[0] is "B"        
	map related functions:  x[0] is "A"
	
```python
# Group by key
rdd = sc.parallelize([(1, "A"), (2, "B"), (3, "C"), (1, "D"), (2, "E"), (1, "F")])
grouped_rdd = rdd.groupByKey()
print([(k, list(v)) for k, v in grouped_rdd.collect()]) # Output: [(1, ['A', 'D', 'F']), (2, ['B', 'E']), (3, ['C'])]

# Reduce by key
reduced_rdd = rdd.reduceByKey(lambda a, b: a + b)
print(reduced_rdd.collect()) # Output: [(1, 'ADF'), (2, 'BE'), (3, 'C')]

# Map values
mapped_values_rdd = rdd.mapValues(lambda x: x.lower()) 
print(mapped_values_rdd.collect()) # Output: [(1, 'a'), (2, 'b'), (3, 'c'), (1, 'd'), (2, 'e'), (1, 'f')]

# Inner Join RDDs 
rdd1 = sc.parallelize([(1, "A"), (2, "B"), (3, "C")])
rdd2 = sc.parallelize([(1, "X"), (2, "Y"), (4, "Z")])
joined_rdd = rdd1.join(rdd2) 
print(joined_rdd.collect())  # Output: [(1, ('A', 'X')), (2, ('B', 'Y'))]

# Full outer join RDDs 
full_outer_joined_rdd = rdd1.fullOuterJoin(rdd2) 
print(full_outer_joined_rdd.collect()) # Output: [(1, ('A', 'X')), (2, ('B', 'Y')), (3, ('C', None)), (4, (None, 'Z'))]

# Left outer join RDDs
left_outer_joined_rdd = rdd1.leftOuterJoin(rdd2)
print(left_outer_joined_rdd.collect())  # Output: [(1, ('A', 'X')), (2, ('B', 'Y')), (3, ('C', None))]

# Right outer join RDDs
right_outer_joined_rdd = rdd1.rightOuterJoin(rdd2)
print(right_outer_joined_rdd.collect())  # Output: [(1, ('A', 'X')), (2, ('B', 'Y')), (4, (None, 'Z'))]

# Union RDDs
union_rdd = rdd1.union(rdd2)
print(union_rdd.collect())  # Output: [(1, 'A'), (2, 'B'), (3, 'C'), (1, 'X'), (2, 'Y'), (4, 'Z')]

# groupWith() example
grouped_rdd = rdd1.groupWith(rdd2)
print([(x, tuple(map(list, y))) for x, y in grouped_rdd.collect()])  
# Output: [(1, (['A'], ['X'])), (2, (['B'], ['Y'])), (3, (['C'], [])), (4, ([], ['Z']))]

# combineByKey() example
combineByKey_rdd = sc.parallelize([(1, 2), (3, 4), (3, 6), (4, 3)])
sum_count_rdd = combineByKey_rdd.combineByKey(
    (lambda x: (x, 1)),  # Create a tuple (value, 1) as the initial value for each key
    (lambda x, y: (x[0] + y, x[1] + 1)),  # Sum the values and increment the count for each key
    (lambda x, y: (x[0] + y[0], x[1] + y[1]))  # Merge the sum and count of each partition
)
print(sum_count_rdd.collect())  # Output: [(1, (2, 1)), (3, (10, 2)), (4, (3, 1))]

# Compute the average value for each key
average_rdd = sum_count_rdd.mapValues(lambda x: x[0] / x[1])
print(average_rdd.collect())  # Output: [(1, 2.0), (3, 5.0), (4, 3.0)]

# lookup() example
rdd = sc.parallelize([(1, "A"), (2, "B"), (3, "C"), (2, "D")])
result = rdd.lookup(2)
print(result)  # Output: ['B', 'D']


# map().reduceByKey() (work on values, kinda similar to value_counts() in pandas df)
reduced_rdd = union_rdd.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)
# union_rdd.map(lambda x: (x[0], 1)) --> [(1, 1), (2, 1), (3, 1), (1, 1), (2, 1), (4, 1)]
print(reduced_rdd.collect())  # Output: [(1, 2), (2, 2), (3, 1), (4, 1)]

# Intersection RDDs
intersection_rdd = rdd1.intersection(rdd2)
print(intersection_rdd.collect())  # Output: []

# Subtract RDDs
rdd1 = sc.parallelize([(1, "A"), (2, "B"), (3, "C")])
rdd2 = sc.parallelize([(1, "X"), (2, "B"), (4, "Z")])
subtract_rdd = rdd1.subtract(rdd2)
print(subtract_rdd.collect())  # Output: [(1, 'A'), (3, 'C')]

# Cartesian product RDDs
rdd1 = sc.parallelize([(1, "A"), (2, "B")])
rdd2 = sc.parallelize([(1, "X"), (2, "Y")])
cartesian_product_rdd = rdd1.cartesian(rdd2)
print(cartesian_product_rdd.collect())  # Output: [((1, 'A'), (1, 'X')), ((1, 'A'), (2, 'Y')), ((2, 'B'), (1, 'X')), ((2, 'B'), (2, 'Y'))]

# Cogroup RDDs 
cogroup_rdd = rdd1.cogroup(rdd2) 
# The content of `cogroup_rdd` is as follows: 
# [(1, (<iterable object1>, <iterable object2>)), (2, (<iterable object3>, <iterable object4>))]
print([(x, (list(y[0]), list(y[1]))) for x, y in cogroup_rdd.collect()]) # Output: [(2, (['B'], ['Y'])), (1, (['A'], ['X']))]
```

### RDD filtering
```python
# Initialize SparkContext
sc = SparkContext("local", "FilteringExamples")

# Sample data: (name, age, country)
data = [
    ("Alice", 30, "USA"),
    ("Bob", 28, "USA"),
    ("Catherine", 25, "UK"),
    ("David", 32, "Canada"),
    ("Eva", 24, "Australia"),
    ("Frank", 45, "USA"),
    ("George", 35, "UK"),
    ("Hannah", 29, "Canada"),
    ("Ian", 31, "Australia"),
    ("Julia", 27, "UK"),
]

# Create an RDD from the sample data
rdd = sc.parallelize(data)

# Filter people who are older than 30
older_than_30 = rdd.filter(lambda x: x[1] > 30)
print(older_than_30.collect()) # Output: [('David', 32, 'Canada'), ('Frank', 45, 'USA'), ('George', 35, 'UK')]

# Filter people who live in the USA
usa_residents = rdd.filter(lambda x: x[2] == "USA")
print(usa_residents.collect()) # Output: [('Alice', 30, 'USA'), ('Bob', 28, 'USA'), ('Frank', 45, 'USA')]

# Filter people with names longer than 4 characters
long_names = rdd.filter(lambda x: len(x[0]) > 4)
print(long_names.collect()) # Output: [('Catherine', 25, 'UK'), ('Hannah', 29, 'Canada')]

# Filter people who are older than 30 and live in the UK
older_than_30_and_uk = rdd.filter(lambda x: x[1] > 30 and x[2] == "UK")
print(older_than_30_and_uk.collect()) # Output: [('George', 35, 'UK')]

# Stop the SparkContext
sc.stop()

```


### Persistence
```python
# Cache the RDD (stores the data in memory)
rdd.cache()

# Unpersist the RDD
rdd.unpersist()
```



### Partitioning
- at least as many partitions as you have cores
- partitionBy(100) is a reasonable place to start for large operations
### Example:  word count and sort
```python
import re
from pyspark import SparkConf, SparkContext

def normalizeWords(text):
    return re.compile(r'\W+', re.UNICODE).split(text.lower())
    # this will remove any punctuations and unicode character

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

input = sc.textFile("file:///sparkcourse/book.txt")
words = input.flatMap(normalizeWords)

wordCounts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
wordCounts.take(5)
    #[('self', 111),
    # ('employment', 75),
    # ('building', 33),
    # ('an', 178),
    # ('internet', 26)]
    
wordCountsSorted1 = wordCounts.sortBy(lambda x: x[1], ascending=False)
wordCountsSorted1.take(5)
    # [('you', 1878), ('to', 1828), ('your', 1420), ('the', 1292), ('a', 1191)]

wordCountsSorted2 = wordCounts.map(lambda x: (x[1], x[0])).sortByKey(ascending=False)
wordCountsSorted2.take(5)
    # [(1878, 'you'), (1828, 'to'), (1420, 'your'), (1292, 'the'), (1191, 'a')]

results = wordCountsSorted2.collect()

type(results)
for result in results[:5]:
    count = str(result[0])
    word = result[1].encode('ascii', 'ignore')
    if (word):
        print(word.decode() + ":\t\t" + count)
    #you:            1878
    #to:             1828
    #your:           1420
    #the:            1292
    #a:              1191

sc.stop()
```


# SQLContext

## Initialize SparkSession
```python
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import *

# Create a SparkSession
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

# End a SparkSession
# spark.stop()
```

## Load RDD's, create DataFrames, register table(for SQL)
#####       parse lines from a file
```python
def mapper(line):
    fields = line.split(',')
    return Row(ID=int(fields[0]), name=str(fields[1].encode("utf-8")), \
               age=int(fields[2]), numFriends=int(fields[3]))

lines = spark.sparkContext.textFile(r"C:\SparkCourse\fakefriends.csv")
people = lines.map(mapper)
type(people) #pyspark.rdd.PipelinedRDD
people.take(3)
    #[Row(ID=0, name="b'Will'", age=33, numFriends=385),
    # Row(ID=1, name="b'Jean-Luc'", age=26, numFriends=2),
    # Row(ID=2, name="b'Hugh'", age=55, numFriends=221)]
    
# Infer the schema, and register the DataFrame as a table to use sql.
schemaPeople = spark.createDataFrame(people).cache()
schemaPeople.createOrReplaceTempView("people")
# (Note: register as a temp view/table because SQL cannot be directly executed on DataFrames,
#  the "people" here is NOT replacing the "people" RDD from the map function, 
#   it will only replace temp view with the same name)
```

## Load DataFrames

#####       common options for the `spark.read.option()` method
	you use chain multiple options if needed
-   "**header**": Set to "true" if the first row in the file contains column names, and "false" if it does not.
    -   `option("header", "true")`

-   "**sep**": Specifies the delimiter used to separate the columns in the file.
    -   `option("sep", "\t")` for tab-separated files
    -   `option("sep", ",")` for comma-separated files

-   "**inferSchema**": Set to "true" to automatically infer the column data types, and "false" to treat all columns as strings.
    -   `option("inferSchema", "true")`

-   "**quote**": Specifies the quote character used to escape delimiters within a field.
    -   `option("quote", '"')`

-   "**escape**": Specifies the escape character used to escape an escape character or quote character.
    -   `option("escape", '\\')`

-   "**nullValue**": Specifies the string that represents a null value in the file.
    -   `option("nullValue", 'NA')

-   "**mode**": Specifies the parsing mode when dealing with corrupted/malformed records.
    -   `option("mode", "PERMISSIVE")` (default): sets all fields to null when a record is corrupted and places the corrupted record in a string column called _corrupt_record.
    -   `option("mode", "DROPMALFORMED")`: ignores the whole corrupted record.
    -   `option("mode", "FAILFAST")`: throws an exception when a corrupted record is encountered.

-   "**dateFormat**": Specifies the date format for parsing date columns.
    -   `option("dateFormat", "yyyy-MM-dd")`

-   "**timestampFormat**": Specifies the timestamp format for parsing timestamp columns.
    -   `option("timestampFormat", "yyyy-MM-dd HH:mm:ss")`

#####       if table has header
```python
people = spark.read.option("header", "true").option("inferSchema", "true")\
    .csv("file:///SparkCourse/fakefriends-header.csv")
type(people) # pyspark.sql.dataframe.DataFrame
    
print("Here is our inferred schema:")
people.printSchema()
    #Here is our inferred schema:
    #root
    # |-- userID: integer (nullable = true)
    # |-- name: string (nullable = true)
    # |-- age: integer (nullable = true)
    # |-- friends: integer (nullable = true)
```

#####       if table has no header, define schema
```python
# from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Define the schema
schema = StructType([
    StructField("userID", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("friends", IntegerType(), True)
])

# Read the CSV file without headers and apply the schema
people2 = spark.read.option("header", "false")\
    .schema(schema)\
    .csv("file:///SparkCourse/fakefriends-noheader.csv") # pyspark.sql.dataframe.DataFrame

print("Here is our manually defined schema:")
people2.printSchema()
    #Here is our inferred schema:
    #root
    # |-- userID: integer (nullable = true)
    # |-- name: string (nullable = true)
    # |-- age: integer (nullable = true)
    # |-- friends: integer (nullable = true)
```

###### common type in StructField
**StringType**: Represents string data (`"helloworld"`, `"3"`, `"example"`).

**IntegerType**: Represents 32-bit integer data (`3`, `2519`, `-100`).

**LongType**: Represents 64-bit integer data (`3000000000`, `-5000000000`, `123456789012345`).

**FloatType**: Represents single-precision floating-point numbers (32-bit) (`3.14`, `-0.5`, `1.414`).

**DoubleType**: Represents double-precision floating-point numbers (64-bit) (`1.7976931348623157E+308`, `-3.14159265359`, `2.718281828459`).

**BooleanType**: Represents boolean values (`True`, `False`).

**TimestampType**: Represents timestamp values (`"2023-03-20 12:34:56"`, `"1969-07-20 20:17:40"`).

**DateType**: Represents date values (`"2023-03-20"`, `"1969-07-20"`).


**ArrayType**: Represents an array of elements (`[1, 2, 3]`, `["apple", "banana", "cherry"]`).

**MapType**: Represents a key-value pair structure (`{"key1": "value1", "key2": "value2"}`, `{"apple": 3, "banana": 2}`).

**StructType**: Represents a complex structure of nested fields (like a JSON object or a DataFrame row) (`{"field1": "value1", "field2": 2}`, `{"name": "John", "age": 30}`).
    
*Note that for complex types (**ArrayType**, **MapType**, and **StructType**), you'll need to define their elements' types as well.


#####       write your own
```python
# Load the data
data = [("Alice", 30, 100),
        ("Bob", 30, 200),
        ("Cathy", 40, 300),
        ("David", 40, 150)]

# Create a DataFrame with schema
schema = ["name", "age", "friends"]
people_df = spark.createDataFrame(data, schema=schema)

# Register the DataFrame as a temporary table
people_df.createOrReplaceTempView("people")
```

#### Check whether 2 spark DataFrames are the same
you **cannot** use `==` because the `==` operator checks if two DataFrame objects reference the same instance, not whether their contents are the same.
use the `df1.subtract(df2).count()` and `df2.subtract(df1).count()`
```python
def are_dataframes_equal(df1, df2):
    diff1 = df1.subtract(df2).count()
    diff2 = df2.subtract(df1).count()
    if diff1 == 0 and diff2 == 0:
        print("DataFrames are equal.")
    else:
        print("DataFrames are not equal.")
        
are_dataframes_equal(dataframe1, dataframe2) 
```

## SQL queries on spark DataFrames
```python

# Select all records
all_records = spark.sql("SELECT * FROM people")
all_records.show()
# +-----+---+-------+
# | name|age|friends|
# +-----+---+-------+
# |Alice| 30|    100|
# |  Bob| 30|    200|
# |Cathy| 40|    300|
# |David| 40|    150|
# +-----+---+-------+

# Filter by age
age_filter = spark.sql("SELECT * FROM people WHERE age = 30")
age_filter.show()
# +-----+---+-------+
# | name|age|friends|
# +-----+---+-------+
# |Alice| 30|    100|
# |  Bob| 30|    200|
# +-----+---+-------+

# Group by and aggregate
group_by_age = spark.sql("SELECT age, COUNT(*) as count, AVG(friends) as avg_friends FROM people GROUP BY age")
group_by_age.show()
# +---+-----+-----------+
# |age|count|avg_friends|
# +---+-----+-----------+
# | 30|    2|      150.0|
# | 40|    2|      225.0|
# +---+-----+-----------+

# Order by
sorted_by_age = spark.sql("SELECT * FROM people ORDER BY age")
sorted_by_age.show()
# +-----+---+-------+
# | name|age|friends|
# +-----+---+-------+
# |Alice| 30|    100|
# |  Bob| 30|    200|
# |Cathy| 40|    300|
# |David| 40|    150|
# +-----+---+-------+
```

## Functions on Spark DataFrames

###### Transformations
`select`, `filter` / `where`, `withColumn`, `withColumnRenamed`, `alias`, `drop`, `dropDuplicates` / `distinct`, `fillna`, `replace`, `sample`, `limit`, `union`, `coalesce`, `repartition`

###### Aggregations and Grouping
`groupBy`, `agg`, `count`, `sum`, `mean` / `avg`, `min`, `max`, `pivot`, `rollup`, `cube`, `countDistinct`, `first`, `last`, `collect_list`, `collect_set`, `corr`, `covar_pop`, `covar_samp`, `cume_dist`, `dense_rank`, `lag`, `lead`, `percent_rank`, `row_number`, `stddev`, `stddev_pop`, `stddev_samp`, `variance`, `var_pop`, `var_samp`, `histogram_numeric`

###### Joins
`join`, `crossJoin`

###### Sorting
`orderBy` / `sort`

###### Miscellaneous
`describe`, `na`, `explain`, `persist` / `cache`, `unpersist`, `printSchema`, `schema`, `selectExpr`, `toJSON`, `toPandas`, `registerTempTable`, `createOrReplaceTempView`, `createGlobalTempView`, `createTempView`, `dropTempView`, `dropGlobalTempView`, `sql`, `hint`, `subtract`, `intersect`, `intersectAll`, `exceptAll`, `crosstab`, `freqItems`, `randomSplit`, `toDF`

### Comprehensive Examples
```python
from pyspark.sql.functions import *

# display columns
people_df.select("name", "friends").show()
    #+-----+-------+
    #| name|friends|
    #+-----+-------+
    #|Alice|    100|
    #|  Bob|    200|
    #|Cathy|    300|
    #|David|    150|
    #+-----+-------+

# groupby, count, orderby
people_df.groupBy("age").count().orderBy("count", ascending=False).show()
# or
people_df.groupBy("age").count().sort(func.col("count").desc()).show()
    #+---+-----+
    #|age|count|
    #+---+-----+
    #| 30|    2|
    #| 40|    1|
    #| 45|    1|
    #+---+-----+


# Make everyone 10 years older
people_df.select(people_df.name, people_df.age + 10).show()
    #+-----+----------+
    #| name|(age + 10)|
    #+-----+----------+
    #|Alice|        40|
    #|  Bob|        40|
    #|Cathy|        50|
    #|David|        55|
    #+-----+----------+
    
    
# Filter by age
people_df.filter(people_df.age == 30).show()
    # +-----+---+-------+
    # | name|age|friends|
    # +-----+---+-------+
    # |Alice| 30|    100|
    # |  Bob| 30|    200|
    # +-----+---+-------+

# Group by and aggregate
from pyspark.sql.functions import avg, count as sql_count    
group_by_age = people_df.groupBy("age").agg(sql_count("*").alias("count"), avg("friends").alias("avg_friends"))
group_by_age.show()
    #+---+-----+-----------+
    #|age|count|avg_friends|
    #+---+-----+-----------+
    #| 30|    2|      150.0|
    #| 40|    1|      300.0|
    #| 45|    1|      150.0|
    #+---+-----+-----------+

# Drop a specific column
people_df.drop("age").show()
    #+-----+-------+
    #| name|friends|
    #+-----+-------+
    #|Alice|    100|
    #|  Bob|    200|
    #|Cathy|    300|
    #|David|    150|
    #+-----+-------+

# Add or replace a column
with_column = people_df.withColumn("age", people_df["age"] * 2)
with_column.show()
    #+-----+---+-------+
    #| name|age|friends|
    #+-----+---+-------+
    #|Alice| 60|    100|
    #|  Bob| 60|    200|
    #|Cathy| 80|    300|
    #|David| 90|    150|
    #+-----+---+-------+

# Rename a column
renamed_column = people_df.withColumnRenamed("friends", "numFriends")
renamed_column.show()
    #+-----+---+----------+
    #| name|age|numFriends|
    #+-----+---+----------+
    #|Alice| 30|       100|
    #|  Bob| 30|       200|
    #|Cathy| 40|       300|
    #|David| 45|       150|
    #+-----+---+----------+

# Get distinct rows
people_df.select("age").distinct().show()
    #+---+
    #|age|
    #+---+
    #| 30|
    #| 40|
    #| 45|
    #+---+

# Drop duplicates
no_duplicates = people_df.dropDuplicates(["age"])
no_duplicates.show()
    #+-----+---+-------+
    #| name|age|friends|
    #+-----+---+-------+
    #|Alice| 30|    100|
    #|Cathy| 40|    300|
    #|David| 45|    150|
    #+-----+---+-------+

# Join two DataFrames
departments = [("HR", "Alice"), ("IT", "Bob"), ("HR", "Cathy"), ("IT", "David"), ("HR", "Eva")]
dept_schema = ["department", "name"]
dept_df = spark.createDataFrame(departments, schema=dept_schema)

joined_df = people_df.join(dept_df, on="name", how="inner")
joined_df.show()
    #+-----+---+-------+----------+
    #| name|age|friends|department|
    #+-----+---+-------+----------+
    #|Alice| 30|    100|        HR|
    #|  Bob| 30|    200|        IT|
    #|Cathy| 40|    300|        HR|
    #|David| 45|    150|        IT|
    #+-----+---+-------+----------+

# Union two DataFrames
more_people = [("Frank", 35, 160), ("Grace", 25, 220)]
more_people_df = spark.createDataFrame(more_people, schema=schema)

union_df = people_df.union(more_people_df)
union_df.show()
    #+-----+---+-------+
    #| name|age|friends|
    #+-----+---+-------+
    #|Alice| 30|    100|
    #|  Bob| 30|    200|
    #|Cathy| 40|    300|
    #|David| 45|    150|
    #|Frank| 35|    160|
    #|Grace| 25|    220|
    #+-----+---+-------+

# Describe (summary statistics)
people_df.describe(["age"]).show()
    #+-------+-----------------+
    #|summary|              age|
    #+-------+-----------------+
    #|  count|                4|
    #|   mean|            36.25|
    #| stddev|7.499999999999998|
    #|    min|               30|
    #|    max|               45|
    #+-------+-----------------+

# Pivot
pivot_df = joined_df.groupBy("department").pivot("age").count()
pivot_df.show()
    #+----------+---+----+----+
    #|department| 30|  40|  45|
    #+----------+---+----+----+
    #|        HR|  1|   1|null|
    #|        IT|  1|null|   1|
    #+----------+---+----+----+

# func.split, func.trim, func.size
from pyspark.sql import functions as func

connections = lines.withColumn("id", func.split(func.trim(func.col("value")), " ")[0]) \
   # create a new column called "id" from reading "value" column, trim leading and trailing white space, split by " ", then take the first element
				   .withColumn("connections", func.size(func.split(func.trim(func.col("value")), " ")) - 1) \
   # create a new column called "connections" from reading "value" column, trim leading and trailing white space, split by " ", then count the total number of elements, then -1


# when().otherwise() function
result = calculateSimilarity \
  .withColumn("score", \
		       func.when(func.col("denominator") != 0, func.col("numerator") / func.col("denominator")) \
		           .otherwise(0) 
              )



```


### UDF 
	UDF (User-Defined Function), a wrapper that allows you to define your own functions and then register it with Spark using the `udf()` function from the `pyspark.sql.functions` module.
#### broadcast variable
	`sparkContext.broadcast()`, a wrapper around the actual data, read-only, cached on each worker node in a Spark cluster.
	you need to use the `.value` attribute on a broadcast object to access the actual data that has been broadcasted.

#### Examples
```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StringType

# Sample data
products_data = {
    1: "Laptop",
    2: "Smartphone",
    3: "Tablet",
}

sales_data = [
    (1, "New York", 1500),
    (2, "Los Angeles", 3000),
    (3, "Chicago", 2000),
    (1, "New York", 1800),
    (2, "Los Angeles", 2800),
    (3, "Chicago", 2200),
]

# Create a SparkSession
spark = SparkSession.builder \
    .appName("UDF and Broadcast Example") \
    .getOrCreate()

# Create DataFrames from the sample data
products_df = spark.createDataFrame(products_data.items(), schema=["productID", "productName"])
sales_df = spark.createDataFrame(sales_data, schema=["productID", "city", "revenue"])

# Broadcast the products data
broadcast_products = spark.sparkContext.broadcast(products_data)

# Define the UDF function
def lookup_product_name(product_id):
    return broadcast_products.value.get(product_id)

# Register the UDF
lookup_product_name_udf = func.udf(lookup_product_name, StringType())

# Use the UDF to map product IDs to their names
sales_with_product_name = sales_df.withColumn("productName", lookup_product_name_udf(func.col("productID")))

# Show the result
sales_with_product_name.show()
    #+---------+-----------+-------+----------+
    #|productID|       city|revenue|productName|
    #+---------+-----------+-------+----------+
    #|        1|   New York|   1500|    Laptop|
    #|        2|Los Angeles|   3000|Smartphone|
    #|        3|    Chicago|   2000|    Tablet|
    #|        1|   New York|   1800|    Laptop|
    #|        2|Los Angeles|   2800|Smartphone|
    #|        3|    Chicago|   2200|    Tablet|
    #+---------+-----------+-------+----------+

# Stop the SparkSession
spark.stop()

```

# Texts Related Processing

### Normalize words
```python
import re

text = "Hello, I'm Jon, ¡Hola! ¿Cómo estás? ¡Estoy bien, gracias!"

words1 = text.split()
print(words1)
# Output: ['Hello,', "I'm", 'Jon,', '¡Hola!', '¿Cómo', 'estás?', '¡Estoy', 'bien,', 'gracias!']

words2 = re.compile(r'\W+', re.UNICODE).split(text)
print(words2)
# Output: ['Hello', 'I', 'm', 'Jon', 'Hola', 'Cómo', 'estás', 'Estoy', 'bien', 'gracias', '']
```

### RDD
```python
import re
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

def normalizeWords(text):
    return re.compile(r'\W+', re.UNICODE).split(text.lower())
    # this will remove any punctuations and unicode character

input = sc.textFile("file:///sparkcourse/book.txt")
words = input.flatMap(normalizeWords)
```

### Spark DataFrames
```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as func

# Read each line of my book into a dataframe
inputDF = spark.read.text("file:///SparkCourse/book.txt")

# Split using a regular expression that extracts words
words = inputDF.select(func.explode(func.split(inputDF.value, "\\W+")).alias("word"))
wordsWithoutEmptyString = words.filter(words.word != "")

# Normalize everything to lowercase
lowercaseWords = wordsWithoutEmptyString.select(func.lower(wordsWithoutEmptyString.word).alias("word"))
```

# Run
### Anaconda prompt
spark-submit file.py


### Check if a spark session is running
```python
isinstance(spark, SparkSession)
# returns `True` or NameError
```
   


