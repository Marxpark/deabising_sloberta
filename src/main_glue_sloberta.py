import os

from GLUE import run_slo_glue

os.makedirs("glue_results", exist_ok=True)
def main():
    glue_results = run_slo_glue.test_model('EMBEDDIA/sloberta',
                                           "/home/mnarat/context-debias/data/glues/SuperGLUE-GoogleMT/csv",
                                           "glue_results/EMBEDDIAsloberta_gmt.txt")
    glue_results2 = run_slo_glue.test_model('EMBEDDIA/sloberta',
                                            "/home/mnarat/context-debias/data/glues/SuperGLUE-HumanT/csv",
                                            "glue_results/EMBEDDIAsloberta_hooman.txt")


if __name__ == "__main__":
    main()

