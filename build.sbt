name := "spark-fm"

version := "1.0"

scalaVersion := "2.11.8"

spName := "hibayesian/spark-fm"

sparkVersion := "2.1.1"

sparkComponents += "mllib"

resolvers += Resolver.sonatypeRepo("public")

//libraryDependencies += "com.typesafe" % "config" % "1.3.2"
// libraryDependencies += "org.slf4j" % "slf4j-api" % "1.7.16"
//libraryDependencies += "com.typesafe" % "config" % "1.2.1"

//unmanagedBase := baseDirectory.value / "custom_lib"

spShortDescription := "spark-fm"

spDescription := """A parallel implementation of factorization machines based on Spark"""
  .stripMargin

//credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")