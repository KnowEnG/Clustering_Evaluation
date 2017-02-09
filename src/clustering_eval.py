"""
@author: The KnowEnG dev team
"""
from clustering_eval_toolbox import clustering_evaluation

def main():
    """
    This is the main function to perform clustering evaluation.
    """
    import sys
    from knpackage.toolbox import get_run_directory_and_file
    from knpackage.toolbox import get_run_parameters

    run_directory, run_file = get_run_directory_and_file(sys.argv)
    run_parameters = get_run_parameters(run_directory, run_file)
    clustering_evaluation(run_parameters)

if __name__ == "__main__":
    main()
