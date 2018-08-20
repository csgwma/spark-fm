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

package org.apache.spark.ml

import java.io.{File, PrintWriter}

import org.apache.spark
import org.apache.spark.mllib.linalg.Vector

package object fm {
  final val PERIOD_K: Int  = 5
  final val WEIGHT_BIAS_FLAG: Int = (1 << 0)
  final val WEIGHT_1WAY_FLAG: Int = (1 << 1)
  final val WEIGHT_2WAY_FLAG: Int = (1 << 2)
  var CONTROL_FLAG: Int = 0
  var WEIGHT_THRESHOLD: Double = 0.0
  var WEIGHT_MINIMUM: Double  = 0.000001
  var COST_RATIO: Double = 1.0

  def saveFmParameters(filePath: String, weights: Vector, dim: (Int, Int, Int), numFeatures: Int): Unit = {
    val writer = new PrintWriter(new File(filePath))
    println(s"# numFeatures=$numFeatures, dim=$dim, weights.size=${weights.size} \n")
    val dV = weights.toDense
    val length = dV.size
    if (dim._1 == 1) {
      writer.write("bias:%d:%.9f\n".format(0, dV(length - 1)))
    }
    if (dim._2 == 1) {
      val wBeg = numFeatures * dim._3
      for (ix <- wBeg until (length - 1)) {
        writer.write("w:%d:%.9f\n".format((ix - wBeg), dV(ix)))
      }
    }
    println("# Write feature vectors")
    if (dim._3 > 0) {
      for (feature <- 0 until numFeatures) {
        writer.write("v:%d:".format(feature))
        for (rank <- 0 until dim._3) {
          val ix = feature * dim._3 + rank
          writer.write("%.9f ".format(dV(ix)))
        }
        writer.write("\n")
      }
    }
    writer.close()
  }

}
