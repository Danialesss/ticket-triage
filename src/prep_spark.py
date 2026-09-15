"""Stage 1 (PySpark): clean raw tickets, add features, split, write Parquet.

Run from the repo root:
    python -m src.prep_spark --input data/sample_tickets.csv --output data/processed

This is deliberately Spark-first. On the tiny sample it is overkill, but the
same script scales to millions of rows on Databricks Community or a cluster,
which is the pattern worth showing on a resume.
"""
import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from src.text_utils import clean_text


def main(input_path: str, output_dir: str, seed: int = 42) -> None:
    spark = SparkSession.builder.appName("ticket-prep").getOrCreate()

    df = spark.read.option("header", True).csv(input_path)
    df = df.dropna(subset=["text", "category"]).dropDuplicates(["text"])

    clean_udf = F.udf(clean_text, StringType())
    df = df.withColumn("clean_text", clean_udf(F.col("text")))
    df = df.filter(F.length("clean_text") > 0)

    df = df.withColumn("word_count", F.size(F.split(F.col("clean_text"), " ")))
    df = df.withColumn("char_count", F.length(F.col("clean_text")))

    train, test = df.randomSplit([0.8, 0.2], seed=seed)
    train.write.mode("overwrite").parquet(os.path.join(output_dir, "train"))
    test.write.mode("overwrite").parquet(os.path.join(output_dir, "test"))

    print(f"train rows: {train.count()}  test rows: {test.count()}")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_tickets.csv")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()
    main(args.input, args.output)
