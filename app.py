from flask import Flask, jsonify
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def getRecommendationBooks():
    # Create a Spark session
    spark = SparkSession.builder.appName("Librari").getOrCreate()

    # Load the ALS model
    model = ALSModel.load("librari_model")
    recommended_user = model.recommendForAllUsers(8)

    filtered = recommended_user.select("recommendations")
    return recommended_user.collect()
