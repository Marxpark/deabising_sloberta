import subprocess
import itertools
import sys
from SEAT import SloSEATAll

commands = ""
def run_training(learning_rate, alpha, beta, level, layer, dir):
    # Construct the command
    global commands
    cmd = [
        "python", "run_debias_mlm.py",
        "--output_dir", "../param_search/" + dir,
        "--model_type", "sloberta",
        "--model_name_or_path", "EMBEDDIA/sloberta",
        "--do_train",
        "--data_file", "../preprocess/42/sloberta/data.bin",
        "--do_eval",
        "--learning_rate", str(learning_rate),
        "--per_gpu_train_batch_size", "16",
        "--per_gpu_eval_batch_size", "16",
        "--num_train_epochs", "10",
        "--block_size", "128",
        "--loss_target", level,
        "--debias_layer", layer,
        "--seed", "66",
        "--evaluate_during_training",
        "--weighted_loss", f"{alpha}", f"{beta}",
        "--dev_data_size", "1000",
        "--square_loss",
        "--line_by_line",
        "--overwrite_output_dir",
        "--gradient_accumulation_steps", "2",
    ]

    commands += ("CUDA_VISIBLE_DEVICES=1 " + " ".join(cmd) + ";\nCUDA_VISIBLE_DEVICES=2 python main.py " + f"\"{dir}\"\n")

    # cmd = "run_debias_mlm.py --output_dir=.../param_search/model --model_type=sloberta --model_name_or_path=EMBEDDIA/sloberta --do_train --data_file=../preprocess/42/sloberta/data.bin --do_eval --learning_rate {} --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --num_train_epochs 15 --block_size 128 --loss_target {} --debias_layer {} --seed 66 --evaluate_during_training --weighted_loss {} --dev_data_size 1000 --square_loss --line_by_line --overwrite_output_dir --gradient_accumulation_steps 2;".format(
    #     str(learning_rate), level, layer, f"{alpha} {beta}"
    # )


# def run_evaluation():
#     # Assuming run_seat() and run_heilman_seat() are executable Python scripts
#     print("RUNING EVALUATION")
#     seat_result = SloSEAT.run_seat("/home/mnarat/context-debias/param_search/model/checkpoint-best")
#     heilman_result = SloSEAT.run_heilman_seat("/home/mnarat/context-debias/param_search/model/checkpoint-best")
#     with open("evaluations.txt", "a") as fl:
#         fl.write("seat_res: {}, heilman_res: {} \n".format(seat_result, heilman_result))
#     return 0,0


# sentence / token
# all / last / first

def main(path):
    print("RUNING EVALUATION")
    print("/home/mnarat/context-debias/param_search/" + path + "/checkpoint-best")
    seat_result = SloSEAT.run_seat("/home/mnarat/context-debias/param_search/" + path + "/checkpoint-best", path)
    heilman_result = SloSEAT.run_heilman_seat("/home/mnarat/context-debias/param_search/" + path + "/checkpoint-best", path)
    with open("evaluations.txt", "a") as fl:
        fl.write("{} -- seat_res: {}, heilman_res: {} \n".format(path, seat_result, heilman_result))


# seat_result = SloSEAT.run_seat()
# heilman_result = SloSEAT.run_heilman_seat()
# with open("sloberta-res.txt", "a") as fl:
#     fl.write("{} -- seat_res: {}, heilman_res: {} \n".format("sloberta", seat_result, heilman_result))
# def write_commands():
#     for target in ["sentence", "token"]:
#         for layer in ["all", "first", "last"]:
#             for lr, alpha in itertools.product(["5e-5", "5e-4", "5e-3", "5e-6", "5e-7"], range(1, 10)):
#                 alpha /= 10
#                 beta = 1 - alpha
#                 run_training(lr, alpha, beta, target, layer)
#     with open("param_search.bash", "w") as fl:
#         fl.write(commands)

def write_commands():
    for target in ["sentence", "token"]:
        for layer in ["all", "first", "last"]:
                run_training("5e-4", 0.3, 0.7, target, layer, target+layer)
    with open("debiasing.bash", "w") as fl:
        fl.write(commands)

if __name__ == "__main__":
    write_commands()


# ORIGINAL RESULTS
# seat [[0.03354388]]
# healiman (0.015728254318237305, 0.00548243370779017, 0.003932063495542142)

# BEST PARAMETERS & RESULTS
# Debias Layer: all, Loss Target: sentence, Learning Rate: 0.0005, Weighted Loss: 0.3 0.7, Heilman Res: (1.5662071526226051e-09, 1.065274497455356e-09, 4.723985127424883e-10), seat res: 0.0
# Debias Layer: first, Loss Target: sentence, Learning Rate: 0.0005, Weighted Loss: 0.6 0.4, Heilman Res: (1.6264458909699716e-09, 3.297278204609682e-10, 8.243195525697265e-10), seat res: 0.0
# Debias Layer: last, Loss Target: sentence, Learning Rate: 5e-05, Weighted Loss: 0.2 0.8, Heilman Res: (3.202164415424559e-10, 1.0811268079361952e-09, 2.5363676879815077e-11), seat res: 0.0
# Debias Layer: all, Loss Target: token, Learning Rate: 0.0005, Weighted Loss: 0.1 0.9, Heilman Res: (1.3315931336300784e-10, 1.6169344870123868e-10, 2.0290943015645118e-10), seat res: 0.0
# Debias Layer: first, Loss Target: token, Learning Rate: 0.0005, Weighted Loss: 0.3 0.7, Heilman Res: (2.650504422560481e-09, 7.989558523043626e-10, 2.0925034612841204e-10), seat res: 0.0
# Debias Layer: last, Loss Target: token, Learning Rate: 5e-05, Weighted Loss: 0.6 0.4, Heilman Res: (9.479674945731873e-10, 2.713913608264033e-09, 3.630176504552438e-10), seat res: 0.0