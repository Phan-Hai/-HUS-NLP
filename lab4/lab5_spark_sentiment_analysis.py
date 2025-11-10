from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

def main():
    print("=" * 60)
    print("Lab 5: Sentiment Analysis with PySpark")
    print("=" * 60)
    
    # Initialize Spark Session
    print("\nInitializing Spark Session...")
    spark = SparkSession.builder \
        .appName("SentimentAnalysis") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark Session initialized")
    
    # CUSTOM DATA PATH
    data_path = r"D:\10. ky1nam4\NLP\data\sentiments.csv"
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"\n✗ ERROR: Data file not found at {data_path}")
        print("Please check:")
        print("  1. File path is correct")
        print("  2. File sentiments.csv exists")
        print("  3. File has 'text' and 'sentiment' columns")
        spark.stop()
        return
    
    print(f"\nLoading data from: {data_path}")
    
    try:
        # Load CSV file
        df = spark.read.csv(data_path, header=True, inferSchema=True)
        
        # Show schema
        print("\nDataset Schema:")
        df.printSchema()
        
        # Show first few rows
        print("\nFirst 5 rows of raw data:")
        df.show(5, truncate=50)
        
        # Check columns
        print(f"\nColumns found: {df.columns}")
        
        # Convert sentiment labels to 0/1 format
        # Assumes: sentiment column contains -1 (negative) and 1 (positive)
        # or 0 (negative) and 1 (positive)
        
        # Check unique sentiment values
        print("\nUnique sentiment values:")
        df.select("sentiment").distinct().show()
        
        # Convert -1/1 to 0/1 if needed
        if df.filter(col("sentiment") == -1).count() > 0:
            print("Converting sentiment from {-1, 1} to {0, 1}...")
            df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
        else:
            print("Using sentiment values as-is for labels...")
            df = df.withColumn("label", col("sentiment").cast("integer"))
        
        # Drop rows with null values
        initial_row_count = df.count()
        df = df.dropna(subset=["text", "sentiment"])
        final_row_count = df.count()
        
        print(f"\n Data loaded successfully")
        print(f"  Initial rows: {initial_row_count}")
        print(f"  After removing nulls: {final_row_count}")
        print(f"  Rows dropped: {initial_row_count - final_row_count}")
        
        # Show label distribution
        print("\nLabel Distribution:")
        df.groupBy("label").count().show()
        
        # Show sample data with labels
        print("\nSample data with labels:")
        df.select("text", "sentiment", "label").show(5, truncate=50)
        
    except Exception as e:
        print(f"\n✗ ERROR loading data: {e}")
        print("\nPlease ensure:")
        print("  1. CSV file has header row")
        print("  2. Columns are named 'text' and 'sentiment'")
        print("  3. File encoding is UTF-8")
        spark.stop()
        return
    
    # Split data into training and test sets
    print("\nSplitting data (80% train, 20% test)...")
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    train_count = train_data.count()
    test_count = test_data.count()
    print(f"  Training samples: {train_count}")
    print(f"  Testing samples: {test_count}")
    
    # Build preprocessing pipeline
    print("\n" + "-" * 60)
    print("Building ML Pipeline")
    print("-" * 60)
    
    # Stage 1: Tokenizer - Split text into words
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    print(" Stage 1: Tokenizer added")
    
    # Stage 2: StopWordsRemover - Remove common stop words
    stopwords_remover = StopWordsRemover(
        inputCol="words", 
        outputCol="filtered_words"
    )
    print(" Stage 2: StopWordsRemover added")
    
    # Stage 3: HashingTF - Convert tokens to feature vectors
    hashing_tf = HashingTF(
        inputCol="filtered_words", 
        outputCol="raw_features", 
        numFeatures=1000
    )
    print(" Stage 3: HashingTF added (numFeatures=1000)")
    
    # Stage 4: IDF - Apply inverse document frequency weighting
    idf = IDF(inputCol="raw_features", outputCol="features")
    print(" Stage 4: IDF added")
    
    # Stage 5: Logistic Regression - Classification model
    lr = LogisticRegression(
        maxIter=10, 
        regParam=0.001, 
        featuresCol="features", 
        labelCol="label"
    )
    print(" Stage 5: LogisticRegression added (maxIter=10, regParam=0.001)")
    
    # Assemble complete pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, lr])
    print("\n Pipeline assembled with 5 stages")
    
    # Train the model
    print("\n" + "-" * 60)
    print("Training Model")
    print("-" * 60)
    print("This may take a few moments...")
    
    try:
        model = pipeline.fit(train_data)
        print(" Model trained successfully")
    except Exception as e:
        print(f"✗ ERROR during training: {e}")
        spark.stop()
        return
    
    # Make predictions
    print("\n" + "-" * 60)
    print("Making Predictions")
    print("-" * 60)
    
    try:
        predictions = model.transform(test_data)
        print(" Predictions completed")
    except Exception as e:
        print(f"✗ ERROR during prediction: {e}")
        spark.stop()
        return
    
    # Show sample predictions
    print("\nSample predictions:")
    predictions.select("text", "label", "prediction", "probability") \
        .show(10, truncate=50)
    
    # Evaluate the model
    print("-" * 60)
    print("Evaluating Model")
    print("-" * 60)
    
    # Calculate Accuracy
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    accuracy = evaluator_accuracy.evaluate(predictions)
    
    # Calculate F1-Score
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="f1"
    )
    f1 = evaluator_f1.evaluate(predictions)
    
    # Calculate Precision
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    precision = evaluator_precision.evaluate(predictions)
    
    # Calculate Recall
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )
    recall = evaluator_recall.evaluate(predictions)
    
    print(f"\n{'='*60}")
    print("PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"{'='*60}")
    
    # Additional statistics
    print("\nPrediction Distribution:")
    predictions.groupBy("prediction").count().orderBy("prediction").show()
    
    print("\nConfusion Matrix (Actual vs Predicted):")
    predictions.groupBy("label", "prediction").count() \
        .orderBy("label", "prediction").show()
    
    # Calculate confusion matrix values
    tp = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
    tn = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
    fp = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
    fn = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()
    
    print("\nDetailed Confusion Matrix:")
    print(f"  True Positives (TP):  {tp}")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    
    # Show some correct and incorrect predictions
    print("\n" + "-" * 60)
    print("Sample Correct Predictions:")
    print("-" * 60)
    correct = predictions.filter(col("label") == col("prediction"))
    correct.select("text", "label", "prediction").show(5, truncate=60)
    
    print("\n" + "-" * 60)
    print("Sample Incorrect Predictions (if any):")
    print("-" * 60)
    incorrect = predictions.filter(col("label") != col("prediction"))
    incorrect_count = incorrect.count()
    if incorrect_count > 0:
        print(f"Total incorrect: {incorrect_count}")
        incorrect.select("text", "label", "prediction").show(5, truncate=60)
    else:
        print("No incorrect predictions! Perfect score!")
    
    print("\n" + "=" * 60)
    print("Spark Sentiment Analysis completed successfully!")
    print("=" * 60)
    
    # Stop Spark session
    spark.stop()
    print("\n Spark session stopped")


if __name__ == "__main__":
    main()