package main.java.com.string.pmml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.sql.SparkSession

object PmmlModel{

  def main(args: Array[String]): Unit = {
    Logger.getLogger("PmmlModel").setLevel(Level.WARN)
    val outputPath = "data/pmml_model1.xml"
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

    testDatadf.describe("age").show()

    testDatadf.stat.crosstab("gender", "smoking_status").show()
    val quantiles = testDatadf.stat.approxQuantile("bmi", Array(0.25, 0.5, 0.95), 0.0)
    println(quantiles.toList)
    testDatadf.groupBy("smoking_status").count().show()
    val count = testDatadf.filter(testDatadf.col("stroke") === 1 && testDatadf.col("gender") === "Female").count()
    println(count.toInt)

    val missingInsertDF = testDatadf.na.fill("missing", Seq("smoking_status"))

    val imputer = new Imputer().setInputCols(Array("bmi"))
      .setOutputCols(Array("bmi_full"))
      .setStrategy("mean")

    val imputedDF = imputer.fit(missingInsertDF).transform(missingInsertDF)
    imputedDF.show()

    val finalDF = imputedDF.drop(imputedDF.col("bmi"))
    finalDF.show()

    val formula = new RFormula().setFormula("stroke ~ age + hypertension + heart_disease + gender + ever_married + work_type + bmi_full").setFeaturesCol("features").setLabelCol("label")

    val formulaDF = formula.fit(finalDF).transform(finalDF)

    formulaDF.show()

    val dtc = new DecisionTreeClassifier()
    val pipeLine = new Pipeline().setStages(Array(formula, dtc))

    val Array(train, test) = finalDF.randomSplit(Array[Double](0.7, 0.3))

    val pipelineModel = pipeLine.fit(train)
    val predictions = pipelineModel.transform(test)
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("stroke").setPredictionCol("prediction").setMetricName("accuracy")
    val values = evaluator.evaluate(predictions)

    println(values)
    val schema = finalDF.schema
    PmmlCreator.getPmmlFile(pipelineModel, schema, outputPath)

  }
}
