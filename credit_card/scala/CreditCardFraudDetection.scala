import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{Imputer, VectorAssembler, StandardScaler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.functions._

object CreditCardFraudDetection {
  
  def main(args: Array[String]): Unit = {
    // Set JAVA_HOME environment variable
    System.setProperty("java.home", "/usr/lib/jvm/java-21-openjdk-amd64")
    
    // Create Spark Session 
    val spark = SparkSession.builder()
      .appName("CreditCardFraudDetection")
      .config("spark.driver.memory", "2g")
      .master("local[*]") // Run locally with all available cores
      .getOrCreate()

    // Set log level to reduce verbose output
    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._
    
    println("Spark session created successfully!")
    println(s"Spark version: ${spark.version}")
    
    try {
      // Load the balanced credit card dataset
      val df = spark.read
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load("/home/aaronpham/Coding/bigdata/spark/spark_mllib/data/creditcard_balanced_single.csv")

      println("Dataset loaded successfully!")
      df.printSchema()
      
      // Show first 5 rows
      println("\nFirst 5 rows of the dataset:")
      df.show(5)
      
      // Dataset statistics
      println(s"\nTotal records: ${df.count()}")
      
      // Check class distribution
      println("\nClass distribution:")
      df.groupBy("Class").count().show()
      
      // Split dataset into train and test (80-20 split)
      val Array(train, test) = df.randomSplit(Array(0.8, 0.2), seed = 42L)
      
      println(s"\nDataset split:")
      println(s"Train count: ${train.count()}")
      println(s"Test count: ${test.count()}")
      
      // Replace null values with mean values using Imputer
      println("\nReplacing null values with mean values...")
      val numericalFeaturesList = train.columns
      
      println(s"Numerical features: ${numericalFeaturesList.mkString(", ")}")
      
      // Create and fit imputer
      val imputer = new Imputer()
        .setInputCols(numericalFeaturesList)
        .setOutputCols(numericalFeaturesList)
        .setStrategy("mean")
      
      val imputerModel = imputer.fit(train)
      
      // Transform train and test datasets
      var trainImputed = imputerModel.transform(train)
      var testImputed = imputerModel.transform(test)
      
      println("Imputation completed. Sample of training data:")
      trainImputed.show(3)
      
      // Aggregate all columns into one features vector
      println("\nCreating feature vector...")
      val inputCols = df.columns.filter(_ != "Class")
      
      println(s"Input columns: ${inputCols.mkString(", ")}")
      
      // Create VectorAssembler
      val assembler = new VectorAssembler()
        .setInputCols(inputCols)
        .setOutputCol("features_assembled")
      
      // Transform datasets
      trainImputed = assembler.transform(trainImputed)
      testImputed = assembler.transform(testImputed)
      
      println("Feature vector assembly completed. Sample:")
      trainImputed.show(2)
      
      // Standardize the dataset
      println("\nStandardizing features...")
      
      // Create StandardScaler
      val scaler = new StandardScaler()
        .setInputCol("features_assembled")
        .setOutputCol("features")
        .setWithStd(true)
        .setWithMean(true)
      
      // Fit scaler on training data
      val scalerModel = scaler.fit(trainImputed)
      
      // Transform both train and test datasets
      trainImputed = scalerModel.transform(trainImputed)
      testImputed = scalerModel.transform(testImputed)
      
      println("Feature standardization completed. Sample:")
      trainImputed.show(3)
      
      // Show sample of features vector
      println("\nSample features vectors:")
      trainImputed.select("features").take(3).foreach(println)
      
      // Create and train Logistic Regression model
      println("\nTraining Logistic Regression model...")
      
      val lr = new LogisticRegression()
        .setFeaturesCol("features")
        .setLabelCol("Class")
        .setMaxIter(100)
        .setRegParam(0.0)
      
      println("Logistic Regression parameters:")
      println(s"Max Iterations: ${lr.getMaxIter}")
      println(s"Regularization Parameter: ${lr.getRegParam}")
      println(s"Features Column: ${lr.getFeaturesCol}")
      println(s"Label Column: ${lr.getLabelCol}")
      
      // Train the model
      val model = lr.fit(trainImputed)
      
      // Make predictions on training data
      val predTrainDf = model.transform(trainImputed)
      
      println("Training predictions sample:")
      predTrainDf.show(5)
      
      // Make predictions on test data
      println("\nMaking predictions on test data...")
      val predTestDf = model.transform(testImputed)
      
      println("Test predictions sample:")
      predTestDf.show(10)
      
      // Model evaluation
      println("\n=== MODEL EVALUATION ===")
      
      // Create evaluators
      val evaluatorAuc = new BinaryClassificationEvaluator()
        .setLabelCol("Class")
        .setMetricName("areaUnderROC")
      
      val evaluatorAcc = new MulticlassClassificationEvaluator()
        .setLabelCol("Class")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
      
      val evaluatorPrecision = new MulticlassClassificationEvaluator()
        .setLabelCol("Class")
        .setPredictionCol("prediction")
        .setMetricName("weightedPrecision")
      
      val evaluatorRecall = new MulticlassClassificationEvaluator()
        .setLabelCol("Class")
        .setPredictionCol("prediction")
        .setMetricName("weightedRecall")
      
      // Calculate metrics
      val auc = evaluatorAuc.evaluate(predTestDf)
      val accuracy = evaluatorAcc.evaluate(predTestDf)
      val precision = evaluatorPrecision.evaluate(predTestDf)
      val recall = evaluatorRecall.evaluate(predTestDf)
      
      // Print results
      println(f"AUC = $auc%.4f")
      println(f"Accuracy = $accuracy%.4f")
      println(f"Precision = $precision%.4f")
      println(f"Recall = $recall%.4f")
      
      // Display model parameters
      println("\n=== MODEL PARAMETERS ===")
      println(s"Intercept: ${model.intercept}")
      println(s"Coefficients: ${model.coefficients}")
      
      // Save results
      println("\nSaving results...")
      import org.apache.spark.sql.functions.col
      
      // Convert probability vector to string for CSV compatibility
      predTestDf.select(
        col("Class"), 
        col("prediction"), 
        col("probability").cast("string").alias("probability_str")
      ).coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv("/home/aaronpham/Coding/bigdata/spark/spark_mllib/results/Classification_Structured_Scala")
      
      println("Results saved successfully to: /home/aaronpham/Coding/bigdata/spark/spark_mllib/results/Classification_Structured_Scala")
      
    } catch {
      case e: Exception =>
        println(s"An error occurred: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      // Stop Spark session
      spark.stop()
      println("Spark session stopped.")
    }
  }
}
