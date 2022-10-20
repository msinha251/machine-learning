import bentoml
from bentoml.io import JSON

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")

model_runner = model_ref.to_runner()

svc = bentoml.Service('mlzoomcamp_homework', runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def classify(application_data):
    prediction = model_runner.predict.run(application_data)
    return { "prediction": prediction }