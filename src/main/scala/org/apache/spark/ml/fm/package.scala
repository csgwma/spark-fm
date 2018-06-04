package org.apache.spark.ml

import java.io.{File, PrintWriter}

import org.apache.spark
import org.apache.spark.mllib.linalg.Vector

import scala.util.Random

package object fm {
  final val RESET_VALUE: Double  = 0.00000001
  final val PERIOD_K: Int  = 5
  final val WEIGHT_BIAS_FLAG: Int = (1 << 0)
  final val WEIGHT_1WAY_FLAG: Int = (1 << 1)
  final val WEIGHT_2WAY_FLAG: Int = (1 << 2)
  var CONTROL_FLAG: Int = 0

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

  def meetReriod(): Boolean = {
    ((new Random).nextInt(1000000) % PERIOD_K) == 0
  }

}
