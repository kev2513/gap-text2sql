import json
import os
import _jsonnet
import os

from seq2struct.commands.infer import Inferer
from seq2struct.datasets.spider import SpiderItem
from seq2struct.utils import registry

import torch

### start - load model

exp_config = json.loads(
    _jsonnet.evaluate_file(
        "experiments/spider-configs/gap-run.jsonnet"))

model_config_path = exp_config["model_config"]
model_config_args = exp_config.get("model_config_args")

infer_config = json.loads(
    _jsonnet.evaluate_file(
        model_config_path,
        tla_codes={'args': json.dumps(model_config_args)}))

infer_config["model"]["encoder_preproc"]["db_path"] = "data/sqlite_files/"

inferer = Inferer(infer_config)

model_dir = exp_config["logdir"] + "/bs=12,lr=1.0e-04,bert_lr=1.0e-05,end_lr=0e0,att=1"
checkpoint_step = exp_config["eval_steps"][0]

model = inferer.load_model(model_dir, checkpoint_step)

### end - load model

from seq2struct.datasets.spider_lib.preprocess.get_tables import dump_db_json_schema
from seq2struct.datasets.spider import load_tables_from_schema_dict
from seq2struct.utils.api_utils import refine_schema_names

def load_db():
    db_id = "singer"
    my_schema = dump_db_json_schema("data/sqlite_files/{db_id}/{db_id}.sqlite".format(db_id=db_id), db_id)
    schema, eval_foreign_key_maps = load_tables_from_schema_dict(my_schema)
    schema.keys()
    dataset = registry.construct('dataset_infer', {
       "name": "spider", "schemas": schema, "eval_foreign_key_maps": eval_foreign_key_maps, 
        "db_path": "data/sqlite_files/"
    })
    for _, schema in dataset.schemas.items():
        model.preproc.enc_preproc._preprocess_schema(schema)
    return dataset.schemas[db_id]

def infer(question, spider_schema):
    data_item = SpiderItem(
            text=None,  # intentionally None -- should be ignored when the tokenizer is set correctly
            code=None,
            schema=spider_schema,
            orig_schema=spider_schema.orig,
            orig={"question": question}
        )
    model.preproc.clear_items()
    enc_input = model.preproc.enc_preproc.preprocess_item(data_item, None)
    preproc_data = enc_input, None
    with torch.no_grad():
        output = inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)
    return output[0]["inferred_code"]

from flask import Flask, request, render_template
from flask_cors import CORS
from sql_formatter.core import format_sql

app = Flask(__name__)
CORS(app)

def postProcessing(prediction, query):
    numbers = [int(s) for s in query.split() if s.isdigit()]
    prediction = format_sql(prediction)
    if numbers:
        prediction = prediction.replace("'terminal'", str(numbers[0]))
    return prediction

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    query = request.form["query"]
    file = request.files["file"]
    if file:
        file.save("data/sqlite_files/singer/singer.sqlite")
    db_schema = load_db()
    columns = ""
    for column in db_schema.columns:
        columns += column.unsplit_name + ", "
    predictionRaw = infer(query, db_schema)
    code = postProcessing(predictionRaw, query)
    return "Query: " + query + "\n\nColumns: " + columns + "\n\n" + code

@app.route("/", methods=['GET', 'POST'])
def index():
    if not request.form:
        return render_template("index.html")
    else:
        prediction = predict()
        return render_template("index.html", prediction = prediction)
