import os

from GLUE import run_slo_glue

os.makedirs("glue_results", exist_ok=True)
def main():

    # base_path= "/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-HumanTmini/csv"
    # base_path_two="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-HumanTmini/csv"

    base_path = "/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-GoogleMT/csv"
    base_path_two = "/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-GoogleMT/csv"

    glue_results = run_slo_glue.test_model('EMBEDDIA/sloberta',
                                           base_path,
                                           "glue_results/EMBEDDIAsloberta_gmt.txt")
    glue_results2 = run_slo_glue.test_model('EMBEDDIA/sloberta',
                                            base_path_two,
                                            "glue_results/EMBEDDIAsloberta_hooman.txt")


if __name__ == "__main__":
    main()

