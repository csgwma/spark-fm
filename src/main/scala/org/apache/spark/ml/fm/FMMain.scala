/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.fm

import scala.io.Source
import scala.collection.mutable.Map
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.optim.configuration.{Algo, Solver}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame

/**
  * The entry for Factorization Machines.
  */
object FMMain {
  val ARG_ALGO = "algo"
  val ARG_SOLVER = "solver"
  val ARG_DIM = "dim"
  val ARG_L1 = "L1"
  val ARG_L2 = "L2"
  val ARG_STDEV = "stdev"
  val ARG_ALPHA = "alpha"
  val ARG_BETA = "beta"
  val ARG_TOL = "tol"
  val ARG_MAX_ITER = "max_iter"
  val ARG_THRESHOLD = "threshold"
  val ARG_MINIBATCHFRATIO = "mini_batch_fraction"
  val ARG_STEP_SIZE = "step_size"
  val ARG_NUM_PARTITION = "num_partitions"
  val ARG_SAVE_PATH = "save_path"
  val ARG_CONTROL_FLAG = "control_flag"
  val ARG_TRAIN_FILE = "train_file"
  val ARG_TEST_FILE = "test_file"
  val ARG_COST_RATIO= "cost_ratio"
  // The types of the arguments
  val TYPE_TRIBLE_INT = Set(ARG_DIM)
  val TYPE_TRIBLE_DOUBLE = Set(ARG_L1, ARG_L2, ARG_ALPHA, ARG_BETA)
  val TYPE_DOUBLE = Set(ARG_STDEV, ARG_TOL, ARG_THRESHOLD, ARG_STEP_SIZE)
  val TYPE_INT = Set(ARG_MAX_ITER, ARG_MINIBATCHFRATIO, ARG_NUM_PARTITION, CONTROL_FLAG)
  val TYPE_STRING = Set(ARG_ALGO, ARG_SOLVER, ARG_SAVE_PATH, ARG_TRAIN_FILE)

  def getModelParameters(args: Array[String]) = {
    var paras = Map[String, String]()
    for (path <- args) {
      val file = Source.fromFile(path)
      for (line <- file.getLines; if (!line.startsWith("#"))) {
        val tokens = line.split("=")
        val (key, value) = (tokens(0).trim, tokens(1).trim)
        paras += (key -> value)
      }
      file.close
    }
    paras
  }

  def str2TribleDouble(str: String) = {
    val values = str.split(",").map(i => i.trim.toDouble)
    (values(0), values(1), values(2))
  }

  def str2TribleInt(str: String) = {
    val values = str.split(",").map(i => i.trim.toInt)
    (values(0), values(1), values(2))
  }

  def initTrainer(paras: Map[String, String]): FactorizationMachines = {
    val trainer = new FactorizationMachines().setAlgo(Algo.fromString(paras(ARG_ALGO)))
      .setSolver(Solver.fromString(paras(ARG_SOLVER)))
      .setDim(str2TribleInt(paras(ARG_DIM)))
      .setRegParamsL2(str2TribleDouble(paras(ARG_L2)))
      .setInitStdev(paras(ARG_STDEV).toDouble)
      .setTol(paras(ARG_TOL).toDouble)
      .setMaxIter(paras(ARG_MAX_ITER).toInt)
      .setThreshold(paras(ARG_THRESHOLD).toDouble)
      .setNumPartitions(paras(ARG_NUM_PARTITION).toInt)
      .setSavePath(paras(ARG_SAVE_PATH))
      .setControlFlag(paras(ARG_CONTROL_FLAG).toInt)
      .setCostRatio(paras(ARG_COST_RATIO).toDouble)

    paras(ARG_SOLVER) match {
      case "pftrl" => trainer.setReParamsL1(str2TribleDouble(paras(ARG_L1)))
        .setAlpha(str2TribleDouble(paras(ARG_ALPHA)))
        .setBeta(str2TribleDouble(paras(ARG_BETA)))
      case "sgd" => trainer.setStepSize(paras(ARG_STEP_SIZE).toDouble)
        .setMiniBatchFraction(paras(ARG_MINIBATCHFRATIO).toDouble)
      case "psgd" => trainer.setStepSize(paras(ARG_STEP_SIZE).toDouble)
          .setAggregationDepth(2)
      case "lbfgs" => trainer.setStepSize(paras(ARG_STEP_SIZE).toDouble)
      case _ => throw new IllegalArgumentException("Invalid arguments %s".format(paras(ARG_SOLVER)))
    }
    trainer
  }

  def main(args: Array[String]): Unit = {
    val paras = getModelParameters(args)
    paras.foreach(println)
    val spark = SparkSession
      .builder()
      .appName("FactorizationMachinesExample")
      .master("local[*]")
      .getOrCreate()

    val trainer = initTrainer(paras)

    val train = spark.read.format("libsvm").load(paras(ARG_TRAIN_FILE))
    val model = trainer.fit(train)
    if (paras.contains(ARG_TEST_FILE) && paras(ARG_TEST_FILE) != "") {
      val test = spark.read.format("libsvm").load(paras(ARG_TEST_FILE))
      val result = model.transform(test)
      val predictionAndLabel = result.select("prediction", "label")
      val evaluator = new RegressionEvaluator().setMetricName("mae")
      println("MAE: " + evaluator.evaluate(predictionAndLabel))
    }
    spark.stop()
  }
}

