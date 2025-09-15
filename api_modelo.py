import joblib
import os
import logging
import jwt
from functools import wraps

from flask import Flask, request, jsonify
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

JWT_SECRET = "JWT_SENHA"
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_modelo")
Base = declarative_base()
DATABASE_URL = "sqlite:///./predictions.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Prediction(Base):

    __tablename__="predictions"
    id = Column(Integer,primary_key=True,autoincrement=True)
    sepal_length = Column(Float,nullable=False)
    sepal_width = Column(Float,nullable=False)
    petal_length = Column(Float,nullable=False)
    petal_width = Column(Float,nullable=False)
    predicted_class = Column(Integer,nullable=False)
    created_at = Column(DateTime,default=datetime.datetime.utcnow)

Base.metadata.create_all(engine)

model = joblib.load("model_iris.pkl")

logger.info("Modelo carregado com sucesso!")     

app = Flask(__name__)

predictions_cache = {}

TEST_USERNAME = "admin"
TEST_PASSWORD = "secret"

def create_token(username):

    payload = {
        "username":username,
        "exp":datetime.datetime.utcnow() +datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }

    token = jwt.encode(payload,JWT_SECRET,algorithm=JWT_ALGORITHM)


    return token

def token_required(f):
    @wraps(f)

    def decorated(*args,**kwargs):
        return f(*args,**kwargs)

    return decorated

@app.route("/predict",methods=["POST"])
@token_required

def predict():

    """
    Endpoint protegidopor token

    Corpo JSON exemplo:

    {"sepal length":5.1,
    "sepal width": 3.2,
    "petal length":1.4,
    "petal width: 0.2
    }
    """

    data = request.get_json(force=True)

    try:
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])

    except:

        logger.error("Dados invalidos",)
        return jsonify({"msg":"Dados invalidos"}),400
    
    features = (sepal_length,sepal_width,petal_length,petal_width)

    if features in predictions_cache:
        predicted_class = predictions_cache[features]

    else:

        input_data = np.array([features])
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])
        predictions_cache[features] = predicted_class

        logger.info("Cache updated")

    db = SessionLocal()
    new_pred = Prediction(    sepal_length=sepal_length,
    sepal_width=sepal_width,
    petal_length=petal_length,
    petal_width=petal_width,
    predicted_class=predicted_class)
    db.add(new_pred)
    db.commit()
    db.close()
    
    return jsonify({"predicted_class": predicted_class})



@app.route("/predictions",methods=["GET"])
@token_required

def list_predictions():

    limit = int(request.args.get("limit",10))
    offset = int(request.args.get("offset",0))


    db = SessionLocal()
    preds = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).offset(offset).all()

    db.close()

    results = []
    for p in preds:
        results.append({"id":p.id,
                       "sepal_length":p.sepal_length,
                       "sepal_width":p.sepal_width,
                       "petal_length":p.petal_length,
                       "petal_width":p.petal_width,
                        "predicted_class":p.predicted_class,
                        "created_at":p.created_at
                       })
        
    return jsonify(results)

if __name__=="__main__":
    app.run(debug=True)



