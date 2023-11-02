import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from sushi_settlement.settlement import Settlement
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

app = Flask(__name__)
CORS(app)

#config
basic_path = os.path.abspath(os.path.dirname(__file__))
workspace = os.path.join(os.path.dirname(basic_path), "workspace_sushi_detection")
CNN_model_path = os.path.join(basic_path, "sushi_settlement/models/cnn_sushi.h5")
MobileNet_model_path = os.path.join(basic_path, "sushi_settlement/models/mobilenet_sushi.h5")
font_path= os.path.join(basic_path, "sushi_settlement/UNSII-2.ttf")

st = Settlement(
    workspace=workspace,
    CNN_model_path=CNN_model_path,
    MobileNet_model_path=MobileNet_model_path,
    font_path=font_path
)



@app.route("/album", methods=["POST"])
def make_album():
    logger.info("get a request")
    input_json = request.json
    logger.info(str(input_json))
    image_dir, user_config = preprocess_input(input_json)
    if image_dir:
        album_path = album.predict(image_dir, user_config)
        logger.info("finish")
        return jsonify({"album_path": album_path})
    else:
        return jsonify({"album_path": None})


def preprocess_input(Input):
    """
    process the input data
    """
    user_config = {}
    image_dir = ""
    if "name-2" in Input:
        user_config["layout_decison_maker"] = {"max_num_pics_in_one_page": int(Input["name-2"])}
    if "background_color12" in Input:
        bg_color = Input["background_color12"]
        r = int(bg_color[1:3], 16)
        g = int(bg_color[3:5], 16)
        b = int(bg_color[5:7], 16)
        user_config["album_maker"] = {"background_color": (r, g, b)}
    if "field-3" in Input:
        bg_extract_switch = "on" if Input["field-3"] == "First" else "off"
        user_config["recognition_server"] = {"background_detection": bg_extract_switch}
    if "uploadedFilePaths" in Input:
        image_dir = Input["uploadedFilePaths"]
    return image_dir, user_config





@app.route('/upload', methods=['POST'])
def upload_file():
    image_path = os.path.join(workspace, datetime.now().strftime("%Y.%m.%d_%H.%M.%S") + ".jpg")
    logger.info(f"pending to save images to {image_path}")
    uploaded_files = request.files.to_dict()
    
    logger.info(uploaded_files)
    for key, file in uploaded_files.items():
        filename = secure_filename(file.filename)
        file.save(image_path)
        logger.info(f"succeessfully save {filename} to filepath {image_path}")
    return jsonify({"path": image_path, "success":1})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)

    # $flask run --host=0.0.0.0 --port=5005