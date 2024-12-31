import os
import sys

from GLUE import run_slo_glue

os.makedirs("glue_results", exist_ok=True)

def main(path):
    print("RUNING glues")
    print("/home/mnarat/context-debias/param_search/" + path + "/checkpoint-best")
    basePath = ""
    glue_results = run_slo_glue.test_model("/home/mnarat/context-debias/param_search/" + path + "/checkpoint-best",
                                           "/home/mnarat/context-debias/data/glues/SuperGLUE-GoogleMT/csv",
                                           f"glue_results/{path}_gmt.txt")
    glue_results2 = run_slo_glue.test_model("/home/mnarat/context-debias/param_search/" + path + "/checkpoint-best",
                                           "/home/mnarat/context-debias/data/glues/SuperGLUE-HumanT/csv",
                                           f"glue_results/{path}_hooman.txt")


if __name__ == "__main__":
    main(sys.argv[1])


