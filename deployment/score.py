import os, glob, json, joblib
import pandas as pd

def init():
    global model
    path = glob.glob(os.path.join(os.environ["AZUREML_MODEL_DIR"], "**", "*.pkl"), recursive=True)[0]
    model = joblib.load(path)

def run(raw_data):
    data = json.loads(raw_data)["input_data"]
    df = pd.DataFrame(data["data"], columns=data["columns"])
    return model.predict(df).tolist()
