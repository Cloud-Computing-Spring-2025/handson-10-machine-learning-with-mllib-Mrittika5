from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    # Fill missing values in TotalCharges
    df = df.withColumn("TotalCharges", when(col("TotalCharges").isNull(), 0).otherwise(col("TotalCharges")))

    # List of categorical columns to index and encode
    categorical_cols = ["gender", "PhoneService", "InternetService"]
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_Index") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=col + "_Index", outputCol=col + "_Vec") for col in categorical_cols]

    # Label indexer for Churn column
    label_indexer = StringIndexer(inputCol="Churn", outputCol="label")

    # Apply indexers
    for indexer in indexers:
        df = indexer.fit(df).transform(df)
    for encoder in encoders:
        df = encoder.fit(df).transform(df)
    df = label_indexer.fit(df).transform(df)

    # Combine feature columns
    feature_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] + [col + "_Vec" for col in categorical_cols]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    final_df = assembler.transform(df).select("features", "label")

    return final_df

# Task 2: Splitting Data and Building a Logistic Regression Model
def train_logistic_regression_model(df):
    # Split data
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # Train logistic regression
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    lr_model = lr.fit(train_data)

    # Predict and evaluate
    predictions = lr_model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print(f"Logistic Regression AUC: {auc:.4f}")

# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df):
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", labelCol="label", outputCol="selectedFeatures")
    result = selector.fit(df).transform(df)
    print("Top 5 features selected using Chi-Square Test:")
    result.select("selectedFeatures", "label").show(5, truncate=False)

# Task 4: Hyperparameter Tuning with Cross-Validation for Multiple Models
def tune_and_compare_models(df):
    # Split data
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

    models = {
        "LogisticRegression": LogisticRegression(featuresCol="features", labelCol="label"),
        "DecisionTree": DecisionTreeClassifier(featuresCol="features", labelCol="label"),
        "RandomForest": RandomForestClassifier(featuresCol="features", labelCol="label"),
        "GBT": GBTClassifier(featuresCol="features", labelCol="label")
    }

    param_grids = {
        "LogisticRegression": ParamGridBuilder()
            .addGrid(models["LogisticRegression"].regParam, [0.01, 0.1])
            .build(),

        "DecisionTree": ParamGridBuilder()
            .addGrid(models["DecisionTree"].maxDepth, [3, 5, 10])
            .build(),

        "RandomForest": ParamGridBuilder()
            .addGrid(models["RandomForest"].numTrees, [10, 20])
            .build(),

        "GBT": ParamGridBuilder()
            .addGrid(models["GBT"].maxIter, [10, 20])
            .build()
    }

    for name, model in models.items():
        print(f"\nTraining and tuning {name}...")
        cv = CrossValidator(estimator=model,
                            estimatorParamMaps=param_grids[name],
                            evaluator=evaluator,
                            numFolds=5)
        cv_model = cv.fit(train_data)
        best_model = cv_model.bestModel
        predictions = best_model.transform(test_data)
        auc = evaluator.evaluate(predictions)
        print(f"Best {name} AUC: {auc:.4f}")
        print(f"Best Params: {best_model.extractParamMap()}")

# Execute tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()
