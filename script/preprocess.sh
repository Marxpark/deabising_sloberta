model_type=$1
data=$2
seed=42
block_size=128
OUTPUT_DIR=../preprocess/$seed/$model_type

rm -r $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

python -u ../src/preprocess.py --input ../data/$data \
                        --stereotypes ../data/stereotype.txt \
                        --attributes ../data/female.txt,../data/male.txt \
                        --output $OUTPUT_DIR \
                        --seed $seed \
                        --block_size $block_size \
                        --model_type $model_type


# bert basic
#./preprocess.sh bert ../data/news-commentary-v15.en
#./preprocess.sh roberta ../data/news-commentary-v15.en
#./preprocess.sh sloberta ../data/gigafida9percent.si
# CUDA_VISIBLE_DEVICES=1 python preprocess.py --input ../data/gigafida9percent.si --stereotypes ../data/poklici.txt --attributes ../data/zenske.txt,../data/moske.txt --output ../preprocess/42/sloberta --seed 42 --block_size 128 --model_type sloberta

# sloberta pycharm
#--input ../data/gigafida9percent.si --stereotypes ../data/poklici.txt --attributes ../data/zenske.txt,../data/moske.txt --output ../preprocess/42/sloberta --seed 42 --block_size 128 --model_type sloberta

# bert pycharm
#--input ../data/news-commentary-v15.en --stereotypes ../data/stereotype.txt --attributes ../data/female.txt,../data/male.txt --output ../preprocess/42/bert --seed 42 --block_size 128 --model_type bert

#screen -s preproc