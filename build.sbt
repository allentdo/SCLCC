name := "SCLCC"

version := "1.0"

scalaVersion := "2.11.11"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze-viz" % "0.13.1",
  "org.apache.spark" % "spark-core_2.11" % "2.2.0",
  "org.apache.spark" % "spark-mllib_2.11" % "2.2.0"
)