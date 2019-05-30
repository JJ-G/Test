package org.test

import org.apache.spark.ml.clustering.KMeans
//import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.SparkSession

object Kmeans {
  def main(args: Array[String]): Unit ={
    val spark = SparkSession
      .builder()
      .appName(s"$this.getClass.getSimpleName")
      .getOrCreate()

    val dataset = spark.read.format("libsvm").load("file:///cos_person/data/sample_kmeans_data.txt")
    val kmeans = new KMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(dataset)

    val predictions = model.transform(dataset)

//    val evaluator = new ClusteringEvaluator()
//    val silhouette = evaluator.evaluate(predictions)
//    println(s"Silhouette with squared euclidean distance = $silhouette")

    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)

    spark.stop()
  }
}
