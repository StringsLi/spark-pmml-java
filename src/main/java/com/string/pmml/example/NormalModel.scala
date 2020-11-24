package main.java.com.string.pmml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object NormalModel{

  def main(args: Array[String]): Unit =  {
    Logger.getLogger("NormalModel").setLevel(Level.WARN)
    val outputPath = "data/normal.model"

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
    testDatadf.stat.crosstab("gender","smoking_status").show()

    val quantiles = testDatadf.stat.approxQuantile("bmi",Array(0.25,0.5,0.95),0.0)
    println(quantiles.toList)

    testDatadf.groupBy("smoking_status").count().show()
    val count=testDatadf.filter(testDatadf.col("stroke")===1 && testDatadf.col("gender") ==="Female").count()
    println(count.toInt)

    val missingInsertDF = testDatadf.na.fill("missing", Seq("smoking_status"))

    val imputer = new Imputer().setInputCols(Array("bmi"))
                               .setOutputCols(Array("bmi_full"))
                               .setStrategy("mean")

    val imputedDF = imputer.fit(missingInsertDF).transform(missingInsertDF)

    val finalDF = imputedDF.drop(imputedDF.col("bmi"))

    val features = finalDF.columns.filterNot(_.contains("id")).filterNot(_.contains("age")).filterNot(_.contains("hypertension"))
        .filterNot(_.contains("heart_disease")).filterNot(_.contains("avg_glucose_level")).filterNot(_.contains("bmi_full")).filterNot(_.contains("stroke"))
    println(features.toList)
    val encodedFeature = features.flatMap{ name =>
      val stringIndexer =new StringIndexer().setInputCol(name).setOutputCol(name+"_index")
      val onehotEncoder = new OneHotEncoderEstimator().setInputCols(Array(name+"_index")).setOutputCols(Array(name+"_vec")).setDropLast(false)
      Array(stringIndexer,onehotEncoder)
    }

    val pipeline = new Pipeline().setStages(encodedFeature)
    val index_model = pipeline.fit(finalDF)
    val transformedDf = index_model.transform(finalDF)
    val vecFeatures = transformedDf.columns.filter(_.contains("vec"))
    val arrayColumn = Array("age","hypertension","heart_disease","avg_glucose_level","bmi_full")
    val arrayTest = vecFeatures ++ arrayColumn
    println(arrayTest.toList)
    val vectorAssembler = new VectorAssembler().setInputCols(arrayTest).setOutputCol("features")
    val dtc  = new DecisionTreeClassifier().setLabelCol("stroke").setFeaturesCol("features")
    val pipelineAssembler = new Pipeline().setStages(Array(vectorAssembler,dtc))
   /* val resultDf = pipelineAssembler.fit(transformedDf).transform(transformedDf)
    resultDf.show()*/

    val Array(trainig,test)=transformedDf.randomSplit(Array[Double](0.7,0.3))
    //val schemaDf = transformedDf.drop(transformedDf.col("features"))
    val model= pipelineAssembler.fit(trainig)
    val predictions = model.transform(test)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("stroke").setPredictionCol("prediction").setMetricName("accuracy")
    val values=   evaluator.evaluate(predictions)

    println(values)

   val tree = model.stages.last.asInstanceOf[DecisionTreeClassificationModel]

    tree.write.save(outputPath)

  }

}
