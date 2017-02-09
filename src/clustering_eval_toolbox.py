"""
@author: The KnowEnG dev team
"""
import os
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
import knpackage.toolbox as kn

class ColumnType(Enum):
    """Two categories of phenotype traits.
    """
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


def run_post_processing_phenotype_clustering_data(run_parameters):
    """This is the clean up function of phenotype data with nans removed.

    Returns:
        output_dict: dictionary with keys to be categories of phenotype data and values
        to be a list of related dataframes.
    """
    from knpackage.toolbox import get_spreadsheet_df
    from collections import defaultdict

    cluster_phenotype_df = get_spreadsheet_df(run_parameters['phenotype_data_full_path'])
    output_dict = defaultdict(list)

    for column in cluster_phenotype_df:
        if(column == 'Cluster_ID'):
            continue
        cur_df = cluster_phenotype_df[['Cluster_ID', column]].dropna(axis=0)
        num_uniq_value = len(cur_df[column].unique())
        if num_uniq_value == 1:
            continue
        if cur_df[column].dtype == object and num_uniq_value >= run_parameters["threshold"]:
            continue
        if num_uniq_value > run_parameters["threshold"]:
            classification = ColumnType.CONTINUOUS
        else:
            classification = ColumnType.CATEGORICAL
        output_dict[classification].append(cur_df)

    return output_dict

def f_oneway(phenotype_df):
    """ Perform a f_oneway test and report the results.

    Parameters:
        phenotype_df: dataframe with two columns with clusters and phenotype trait values.
        ret: result of the phenotype dataframe.
    """
    groups = []
    uniq_cm_vals = sorted(set(phenotype_df.values[:, 0]))

    phenotype_name = phenotype_df.columns.values[1]
    for i in uniq_cm_vals:
        groups.append(
            phenotype_df.loc[phenotype_df['Cluster_ID'] == i, phenotype_name].values.tolist())

    fval, pval = stats.f_oneway(*groups)
    uniq_trait = np.unique(phenotype_df.values[:, 1].reshape(-1))
    ret = [len(uniq_trait), 'f_oneway', fval, pval]
    return ret

def chisquare(phenotype_df):
    """ Perform a chi-square test and report the results.

    Parameters:
        phenotype_df: dataframe with two columns with clusters and phenotype trait values.
        ret: result of the phenotype dataframe.
    """

    unique_category = np.unique(phenotype_df.values[:, 1])
    unique_cluster = np.unique(phenotype_df.values[:, 0])
    num_clusters = len(unique_cluster)
    num_phenotype = len(unique_category)
    phenotype_name = phenotype_df.columns.values[1]
    phenotype_val_dict = dict(zip(unique_category, range(num_phenotype)))
    cluster_dict = dict(zip(unique_cluster, range(num_clusters)))
    
    cont_table = np.zeros((num_clusters, num_phenotype))
    phenotype_total = np.zeros(num_phenotype)
    cluster_total = np.zeros(num_clusters)
    total_total = 0
    
    for sample in phenotype_df.index:
        clus = cluster_dict[phenotype_df.loc[sample, 'Cluster_ID']]
        trt = phenotype_val_dict[phenotype_df.loc[sample, phenotype_name]]  # pylint: disable=no-member
        cont_table[clus, trt] += 1
        phenotype_total[trt] += 1
        cluster_total[clus] += 1
        total_total += 1
        
    cluster_total_pct = 1.0 * cluster_total / total_total
    phenotype_total_pct = 1.0 * phenotype_total / total_total
    expect = (np.array([cluster_total_pct]).T * phenotype_total)  # pylint: disable=no-member
    dd = (cont_table.shape[0] - 1) * (cont_table.shape[1] - 1)
    chi, pval = stats.chisquare(cont_table, expect, ddof=dd, axis=None)
    ret = [num_phenotype, 'chisquare', chi, pval]
    return ret


def clustering_evaluation(run_parameters):
    """ Run clustering evaluation on the whole dataframe of phenotype data.
    Save the results to tsv file.
    """
    output_dict = run_post_processing_phenotype_clustering_data(run_parameters)
    result_df = pd.DataFrame(index=['Trait_length_after_dropna', 'measure', 'chi/fval', 'pval'])

    for key, df_list in output_dict.items():
        if(key == ColumnType.CATEGORICAL):
            for item in df_list:
                phenotype_name = item.columns.values[1]
                result_df[phenotype_name] = chisquare(item)
        else:
            for item in df_list:
                phenotype_name = item.columns.values[1]
                result_df[phenotype_name] = f_oneway(item)


    file_name = kn.create_timestamped_filename("clustering_evaluation_result", "tsv")
    file_path = os.path.join(run_parameters["results_directory"], file_name)
    result_df.T.to_csv(file_path, header=True, index=True, sep='\t')

