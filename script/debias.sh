model_type=$1
gpu=$2
debias_layer=all # first last all
loss_target=token # token sentence
dev_data_size=1000
seed=42
alpha=0.2
beta=0.8

if [ $model_type = 'bert' ]; then
    model_name_or_path=bert-base-uncased
elif [ $model_type = 'roberta' ]; then
    model_name_or_path=roberta-base
elif [ $model_type = 'albert' ]; then
    model_name_or_path=albert-base-v2
elif [ $model_type = 'dbert' ]; then
    model_name_or_path=distilbert-base-uncased
elif [ $model_type = 'electra' ]; then
    model_name_or_path=google/electra-small-discriminator
elif [ $model_type = 'sloberta' ]; then
    model_name_or_path=EMBEDDIA/sloberta
fi

TRAIN_DATA=../preprocess/$seed/$model_type/data.bin
OUTPUT_DIR=../debiased_models/$seed/$model_type

rm -r $OUTPUT_DIR

echo $model_type $seed

CUDA_VISIBLE_DEVICES=$gpu python ../src/run_debias_mlm.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=$model_type \
    --model_name_or_path=$model_name_or_path \
    --do_train \
    --data_file=$TRAIN_DATA \
    --do_eval \
    --learning_rate 5e-5 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --num_train_epochs 3 \
    --block_size 128 \
    --loss_target $loss_target \
    --debias_layer $debias_layer \
    --seed $seed \
    --evaluate_during_training \
    --weighted_loss $alpha $beta \
    --dev_data_size $dev_data_size \
    --square_loss \
    --line_by_line


# debias layer [all, first, last]
# loss target [sentence, token]


# bert standard
#./debias.sh bert

# sloberta standard
# python run_debias_mlm.py --output_dir=../debiased_models/42/sloberta --model_type=sloberta --model_name_or_path=EMBEDDIA/sloberta --do_train --data_file=../preprocess/42/sloberta/data.bin --do_eval --learning_rate 5e-5 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --num_train_epochs 3 --block_size 128 --loss_target token --debias_layer all --seed 42 --evaluate_during_training --weighted_loss 0.2 0.8 --dev_data_size 1000 --square_loss --line_by_line
# python run_debias_mlm.py --output_dir=../debiased_models/42/sloberta --model_type=sloberta --model_name_or_path=EMBEDDIA/sloberta --do_train --data_file=../preprocess/42/sloberta/data.bin --do_eval --learning_rate 5e-5 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --num_train_epochs 3 --block_size 128 --loss_target token --debias_layer first --seed 42 --evaluate_during_training --weighted_loss 0.2 0.8 --dev_data_size 1000 --square_loss --line_by_line
# python run_debias_mlm.py --output_dir=../debiased_models/42/sloberta --model_type=sloberta --model_name_or_path=EMBEDDIA/sloberta --do_train --data_file=../preprocess/42/sloberta/data.bin --do_eval --learning_rate 5e-5 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --num_train_epochs 3 --block_size 128 --loss_target token --debias_layer last --seed 42 --evaluate_during_training --weighted_loss 0.2 0.8 --dev_data_size 1000 --square_loss --line_by_line

#CUDA_VISIBLE_DEVICES=2 python run_debias_mlm.py --output_dir=../debiased_models/42/sloberta --model_type=sloberta --model_name_or_path=EMBEDDIA/sloberta --do_train --data_file=../preprocess/42/sloberta/data.bin --do_eval --learning_rate 5e-5 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --num_train_epochs 3 --block_size 128 --loss_target token --debias_layer all --seed 42 --evaluate_during_training --weighted_loss 0.2 0.8 --dev_data_size 1000 --square_loss --line_by_line
# sloberta pycharm
#--output_dir=../debiased_models/42/sloberta --model_type=sloberta --model_name_or_path=EMBEDDIA/sloberta --do_train --data_file=../preprocess/42/sloberta/data.bin --do_eval --learning_rate 5e-5 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --num_train_epochs 3 --block_size 128 --loss_target token --debias_layer all --seed 42 --evaluate_during_training --weighted_loss 0.2 0.8 --dev_data_size 1000 --square_loss --line_by_line

#bert pycharm
#--output_dir=../debiased_models/42/bert --model_type=bert --model_name_or_path=bert-base-uncased --do_train --data_file=../preprocess/42/bert/data.bin --do_eval --learning_rate 5e-5 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --num_train_epochs 3 --block_size 128 --loss_target token --debias_layer all --seed 42 --evaluate_during_training --weighted_loss 0.2 0.8 --dev_data_size 1000 --square_loss --line_by_line

# roberta pycharm
#--output_dir=../debiased_models/42/roberta --model_type=roberta --model_name_or_path=roberta-base --do_train --data_file=../preprocess/42/roberta/data.bin --do_eval --learning_rate 5e-5 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --num_train_epochs 3 --block_size 128 --loss_target token --debias_layer all --seed 42 --evaluate_during_training --weighted_loss 0.2 0.8 --dev_data_size 1000 --square_loss --line_by_line