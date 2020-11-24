package main.java.com.string.pmml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{Imputer, VectorAssembler}
import org.apache.spark.sql.SparkSession

object PmmlModel2{

  def main(args: Array[String]): Unit = {
    Logger.getLogger("PmmlModel2").setLevel(Level.WARN)
    val outputPath = "data/pmml_model2.xml"
    val spark = SparkSession.builder()
      .master("local")
      .appName("MeritKMeansTest")
      .config("spark.sql.warehouse.dir", "file:///E:/spark-warehouse")
      .getOrCreate()

    val testDatadf = spark.read
                          .format("csv")
                          .option("delimiter", ",")
                          .option("inferSchema", "true")
                          .option("header", "true")
                          .load("data/train_2v.csv")

    testDatadf.stat.crosstab("gender","smoking_status").show()
    testDatadf.groupBy("smoking_status").count().show()
    val missingInsertDF = testDatadf.na.fill("missing", Seq("smoking_status"))
    val imputer = new Imputer().setInputCols(Array("bmi"))
      .setOutputCols(Array("bmi_full"))
      .setStrategy("mean")
    val imputedDF=imputer.fit(missingInsertDF).transform(missingInsertDF)
    val finalDF = imputedDF.drop(imputedDF.col("bmi"))

    val vectorAsCols = Array("age","hypertension","heart_disease","bmi_full")
    val vectorAssembler = new VectorAssembler().setInputCols(vectorAsCols).setOutputCol("vectorFeature")

    val dtc  = new DecisionTreeClassifier().setFeaturesCol("vectorFeature").setLabelCol("stroke")
    val pipeLine = new Pipeline().setStages(Array(vectorAssembler,dtc))
    val Array(trainig,test)=finalDF.randomSplit(Array[Double](0.7,0.3))
    val pipelineModel = pipeLine.fit(trainig)
    val predictions =pipelineModel.transform(test)
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("stroke").setPredictionCol("prediction").setMetricName("accuracy")
    val values=   evaluator.evaluate(predictions)

    println("测试集的正确率为：" + values)
    val schema = finalDF.schema
    PmmlCreator.getPmmlFile(pipelineModel,schema,outputPath)
  }

  }
