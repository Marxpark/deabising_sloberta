import json
import os

from SEAT import SentAll, SloSEATAll
from GLUE import CustomGLUE

models_names = [
    "sloberta_all_layers_sentence",
    "sloberta_all_layers_token",
    "sloberta_first_layer_sentence",
    "sloberta_first_layer_token",
    "sloberta_last_layer_sentence",
    "sloberta_last_layer_token"
]

debias_strength = [
    "alphaBeta3_7",
    "alphaBeta5_5",
    "alphaBeta7_3",
    "alphaBeta9_1"
]

results_dir = "all_results"

path_to_layers = "/home/mnarat/context-debias/debiased_models/{}/{}/checkpoint-best"


def run_for_all_models():
    save_path = "SLOBERTA"
    path_to_model = "EMBEDDIA/sloberta"
    sentiment = SentAll.run_senti_analaysis(path_to_model, save_path)
    seat = SloSEATAll.run_seat(path_to_model, save_path)
    heilman = SloSEATAll.run_heilman_seat(path_to_model, save_path)
    glue_base = "/home/mnarat/context-debias/data/humanTranslation/"
    glue = CustomGLUE.run_tasks(path_to_model, glue_base, save_path)
    results = {
        "sentiment": sentiment,
        "seat": seat,
        "heilman": heilman,
        "glue": glue
    }

    os.makedirs(results_dir, exist_ok=True)

    # Save results to a JSON file
    results_file = os.path.join(results_dir, "{}-{}".format(save_path, save_path))
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    for strength in debias_strength:
        for model_name in models_names:
            save_path = "{}_{}".format(strength, model_name)
            path_to_model = path_to_layers.format(strength, model_name)
            sentiment = SentAll.run_senti_analaysis(path_to_model, save_path)
            seat = SloSEATAll.run_seat(path_to_model, save_path)
            heilman = SloSEATAll.run_heilman_seat(path_to_model, save_path)
            glue = CustomGLUE.run_tasks(path_to_model, save_path)
            model_results = {
                "sentiment": sentiment,
                "seat": seat,
                "heilman": heilman,
                "glue": glue
            }
            model_results_file = os.path.join(results_dir, "{}-{}".format(strength, model_name))
            with open(model_results_file, "w") as f:
                json.dump(model_results, f, indent=4)
