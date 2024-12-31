import os


glue_tests = [
 'BoolQ/',
    'CB/',
    'COPA/',
    'MultiRC/',
    'ReCoRD/',
    'RTE/',
    'WSC/'
]

dirs = ["11",
        "12",
        "13",
        "14",
        "15",
        "21",
        "22",
        "23",
        "24",
        "25",
        "31",
        "32",
        "33",
        "34",
        "35",
        "41",
        "42"
        "43",
        "44",
        "45",
        "51",
        "52",
        "53",
        "54",
        "55"]
models = [
'sloberta_all_layers',
'sloberta_last_layer',
'sloberta_first_layer',
'sloberta_all_layers_sentence',
'sloberta_last_layer_sentence',
'sloberta_first_layer_sentence'
]
subfolders = [f.path for f in os.scandir('debiased_models') if f.is_dir()]
print(subfolders)

for dir in dirs:
    for model in models:
        for test in glue_tests:
            print("CUDA_VISIBLE_DEVICES=0 python run_glue.py --model_name_or_path {} --train_file {} --validation_file {} --test_file {} --output_dir ./GLUEOUTPUT/result/{} --do_train --do_eval --do_predict --max_seq_length 64 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate 2e-5 --num_train_epochs 1.0 --overwrite_output_dir --max_predict_samples 1000 --max_eval_samples 1000 --max_train_samples 1000".format(
                '/home/mnarat/context-debias/debiased_models/' + dir + '/' + model + '/checkpoint-best',
                '/home/mnarat/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/' + test + 'train.jsonl',
                '/home/mnarat/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/' + test + 'val.jsonl',
                '/home/mnarat/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/' + test + 'test.jsonl',
                'debiased_models/' + dir + '/' + model + '/' + test,
            ))
            print('cp ' + '/home/mnarat/context-debias/debiased_models/' + dir + '/' + model + '/res/log.log ./GLUEOUTPUT/result/{}'.format('debiased_models/' + dir + '/' + model + '/' + test + '/hlm.log'))


# CUDA_VISIBLE_DEVICES=0 python run_glue.py
# --model_name_or_path /home/mnarat/context-debias/debiased_models/11/sloberta_all_layers/checkpoint-best
# --train_file      /home/mnarat/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/BoolQ/train.jsonl
# --validation_file /home/mnarat/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/BoolQ/val.jsonl
# --test_file       /home/mnarat/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/BoolQ/test.jsonl
# --output_dir ./GLUEOUTPUT/result/11/sloberta_all_layers
# --do_train --do_eval --do_predict --max_seq_length 64
# --per_device_train_batch_size 32 --per_device_eval_batch_size 32
# --learning_rate 2e-5 --num_train_epochs 1.0 --overwrite_output_dir
# --max_predict_samples 1000 --max_eval_samples 1000 --max_train_samples 1000
# CUDA_VISIBLE_DEVICES=0 python run_glue.py --model_name_or_path /home/mnarat/context-debias/debiased_models/11/sloberta_all_layers/checkpoint-best --train_file /home/mnarat/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/RTE/train.jsonl --validation_file /home/mnarat/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/RTE/val.jsonl --test_file /home/mnarat/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/RTE/test.jsonl --output_dir ./GLUEOUTPUT/result/debiased_models/11/sloberta_all_layers/RTE/ --do_train --do_eval --max_seq_length 64 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate 2e-5 --num_train_epochs 1.0 --overwrite_output_dir --max_predict_samples 1000 --max_eval_samples 1000 --max_train_samples 1000
