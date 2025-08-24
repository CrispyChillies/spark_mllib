from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, col
import os

# Initialize Spark Session
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"

spark = (
    SparkSession.builder.appName("CreditCardUndersampling")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)


def analyze_dataset(df, class_col="Class"):
    """
    Analyze the original dataset distribution
    """
    print("=" * 50)
    print("ORIGINAL DATASET ANALYSIS")
    print("=" * 50)

    total_records = df.count()
    print(f"Total records: {total_records:,}")

    # Class distribution
    distribution = df.groupBy(class_col).count().orderBy(class_col)
    distribution.show()

    # Calculate imbalance ratio
    fraud_count = df.filter(col(class_col) == 1).count()
    normal_count = df.filter(col(class_col) == 0).count()

    imbalance_ratio = normal_count / fraud_count if fraud_count > 0 else 0

    print(
        f"Normal transactions (Class 0): {normal_count:,} ({normal_count/total_records*100:.2f}%)"
    )
    print(
        f"Fraud transactions (Class 1): {fraud_count:,} ({fraud_count/total_records*100:.2f}%)"
    )
    print(f"Imbalance ratio: 1:{imbalance_ratio:.1f}")

    return fraud_count, normal_count


def perform_undersampling(df, minority_count, class_col="Class", seed=42):
    """
    Perform undersampling to balance the dataset
    """
    print("\n" + "=" * 50)
    print("PERFORMING UNDERSAMPLING")
    print("=" * 50)

    # Get all minority class samples (fraud transactions)
    minority_data = df.filter(col(class_col) == 1)
    print(f"Keeping all {minority_count} fraud transactions")

    # Random sample from majority class
    majority_data_sampled = (
        df.filter(col(class_col) == 0).orderBy(rand(seed=seed)).limit(minority_count)
    )

    print(f"Randomly sampling {minority_count} normal transactions from majority class")

    # Combine to create balanced dataset
    balanced_df = minority_data.union(majority_data_sampled)

    # Verify balanced dataset
    print(f"\nBalanced dataset created with {balanced_df.count()} total records")

    print("\nBalanced dataset distribution:")
    balanced_df.groupBy(class_col).count().orderBy(class_col).show()

    return balanced_df


def save_balanced_dataset(balanced_df, output_path):
    """
    Save the balanced dataset to CSV file
    """
    print("\n" + "=" * 50)
    print("SAVING BALANCED DATASET")
    print("=" * 50)

    # Save as single CSV file
    balanced_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(
        output_path
    )

    print(f"Balanced dataset saved to: {output_path}")
    print(
        "Note: Spark saves as a directory with part files. The CSV file will be inside this directory."
    )

    # Also save as a single file with proper naming
    single_file_path = output_path.replace(".csv", "_single.csv")
    balanced_df.toPandas().to_csv(single_file_path, index=False)
    print(f"Single CSV file saved to: {single_file_path}")


def main():
    """
    Main function to execute undersampling process
    """
    print("CREDIT CARD FRAUD DETECTION - DATASET UNDERSAMPLING")
    print("=" * 60)

    # Read original credit card dataset
    input_path = "/home/aaronpham/Coding/bigdata/spark/spark_mllib/data/creditcard.csv"
    print(f"Loading dataset from: {input_path}")

    try:
        df = spark.read.format("csv").load(input_path, header=True, inferSchema=True)

        print("Dataset loaded successfully!")

        # Analyze original dataset
        fraud_count, normal_count = analyze_dataset(df)

        # Perform undersampling
        balanced_df = perform_undersampling(df, fraud_count)

        # Save balanced dataset
        output_path = "/home/aaronpham/Coding/bigdata/spark/spark_mllib/data/creditcard_balanced.csv"
        save_balanced_dataset(balanced_df, output_path)

        # Final summary
        print("\n" + "=" * 50)
        print("UNDERSAMPLING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Original dataset: {normal_count + fraud_count:,} records")
        print(f"Balanced dataset: {balanced_df.count():,} records")
        print(
            f"Reduction ratio: {((normal_count + fraud_count) / balanced_df.count()):.1f}:1"
        )
        print("\nThe balanced dataset is now ready for machine learning!")
        print("You and your colleagues can use the saved CSV file for training models.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check the input file path and try again.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
