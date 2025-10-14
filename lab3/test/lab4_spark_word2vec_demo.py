import re, os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec, Tokenizer
from pyspark.sql.functions import col, lower, regexp_replace, split

def main():
    spark = SparkSession.builder \
        .appName("Word2Vec Spark Demo") \
        .getOrCreate()

    dataset_path = r"D:\10. ky1nam4\NLP\data\c4-train.00000-of-01024-30K.json"
    df = spark.read.json(dataset_path)
    df = df.select("text").where(col("text").isNotNull())

    df_clean = df.withColumn("text_clean", lower(col("text")))
    df_clean = df_clean.withColumn(
        "text_clean",
        regexp_replace(col("text_clean"), r"[^a-zA-Z0-9\s]", "")
    )
    tokenizer = Tokenizer(inputCol="text_clean", outputCol="words")
    df_words = tokenizer.transform(df_clean)

    word2Vec = Word2Vec(vectorSize=100, minCount=5, inputCol="words", outputCol="model")
    model = word2Vec.fit(df_words)

    synonyms = model.findSynonyms("computer", 5)
    print("5 từ đồng nghĩa với 'computer':")
    synonyms.show()

    save_path = r"D:\10. ky1nam4\NLP\results\word2vec_spark_model.model"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    model.write().overwrite().save(save_path)
    print(f"Model đã được lưu tại: {save_path}")

    spark.stop()

if __name__ == "__main__":
    main()
