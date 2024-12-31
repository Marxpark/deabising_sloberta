import sys

from SEAT import SentAll


def main(path):
    print("RUNING EVALUATION")
    print("/home/mnarat/context-debias/param_search/" + path + "/checkpoint-best")
    if (path.startswith("EMBEDDIA")):
        sent = Sent.run_senti_analaysis()
    else:
        sent = Sent.run_senti_analaysis("/home/mnarat/context-debias/param_search/" + path + "/checkpoint-best", path)


if __name__ == "__main__":
    main(sys.argv[1])
