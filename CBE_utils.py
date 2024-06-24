###
###
###

### Functions used for single cell profile preprocessing and aggregation


### imports
from scipy import stats
import pandas as pd
import pycytominer
import numpy as np
import time
from datetime import date
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from pyod.models.hbos import HBOS
import random
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

###########################################

##### Data Loading

### reduce size of single cell df
## to save memory and prevent crashing of the node we need to reduce size of our df
def optimize_size(df: pd.DataFrame) -> pd.DataFrame:
    ## floats
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    
    ##intigers
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df


### Load profiles and concatenate measurements from different cellular compartments
## now we load and reduce the size of each df before putting them all together

def load_data_CP_Cluster(Path_Nuclei, Path_Cells, Path_Cyto):
    """
    loads data from CellProfiler output, uses chunk loading with chunksize 50000
    :param Path_Nuclei: path to the file Nuclei.csv
    :param Path_Cells: path to the file Cells.csv
    :param Path_Cyto: path to the file Cytoplasm.csv
    :return: Concatenated raw single cell data from CellProfiler from all output files with Meta data
    """
      
    ##### load Nuclei data
    chunk = pd.read_csv(Path_Nuclei, delimiter = ",", skiprows = 0, chunksize = 50000) 
    Nuc = pd.concat(chunk)
    
    Nuc.columns = Nuc.columns.str.strip() ### strips leading white spaces in indexes
    Nuc = Nuc.add_prefix('Nuc_') ### adds Nuc in front of all column names
    Nuc.rename(columns={'Nuc_ImageNumber':'ImageNumber',
                        'Nuc_ObjectNumber':'ObjectNumber',
                        'Nuc_Metadata_Batch':'Metadata_Batch',
                        'Nuc_Metadata_Plate':'Metadata_Plate',
                        'Nuc_Metadata_Well':'Metadata_Well',
                        'Nuc_Metadata_Site':'Metadata_Site'}, inplace=True) ### changing back ImageNumber, ObjectNumber and Metadata
    ### downcast
    Nuc = optimize_size(Nuc)
           
    ##### load Cell data
    chunk = pd.read_csv(Path_Cells, delimiter = ",", skiprows = 0, chunksize = 50000) 
    Cells = pd.concat(chunk)
    
    Cells.columns = Cells.columns.str.strip() ### strips leading white spaces in indexes
    Cells = Cells.add_prefix('Cells_') ### adds Cells_ in front of all column names
    Cells.rename(columns={'Cells_ImageNumber':'ImageNumber',
                          'Cells_ObjectNumber':'ObjectNumber',
                        'Cells_Metadata_Batch':'Metadata_Batch',
                        'Cells_Metadata_Plate':'Metadata_Plate',
                        'Cells_Metadata_Well':'Metadata_Well',
                        'Cells_Metadata_Site':'Metadata_Site'}, inplace=True) ### changing back ImageNumber, ObjectNumber and Metadata
    ### downcast
    Cells = optimize_size(Cells)
    
    ##### load Cytoplasm data
    chunk = pd.read_csv(Path_Cyto, delimiter = ",", skiprows = 0, chunksize = 50000) 
    Cyto = pd.concat(chunk)
    
    Cyto.columns = Cyto.columns.str.strip() ### strips leading white spaces in indexes
    Cyto = Cyto.add_prefix('Cyto_') ### adds Cyto_ in front of all column names
    Cyto.rename(columns={'Cyto_ImageNumber':'ImageNumber',
                         'Cyto_ObjectNumber':'ObjectNumber',
                        'Cyto_Metadata_Batch':'Metadata_Batch',
                        'Cyto_Metadata_Plate':'Metadata_Plate',
                        'Cyto_Metadata_Well':'Metadata_Well',
                        'Cyto_Metadata_Site':'Metadata_Site'}, inplace=True) ### changing back ImageNumber, ObjectNumber and Metadata
    ### downcast
    Cyto = optimize_size(Cyto)
    
    ###
    ###
    ##### joining Nuc, Cells and Cyto data
    Data_Temp = pd.merge(
        Nuc, 
        Cells,
        on = ['Metadata_Batch',
              'Metadata_Plate',
              'Metadata_Well',
              'Metadata_Site',
             'ImageNumber',
             'ObjectNumber'], 
        how = "inner")

    Data = pd.merge(
        Data_Temp, 
        Cyto,
        on = ['Metadata_Batch',
              'Metadata_Plate',
              'Metadata_Well',
              'Metadata_Site',
             'ImageNumber',
             'ObjectNumber'], 
        how = "inner")
    
    ###
    ###### housekeeping
    ### remove columns that are only NaN, should actually not be the case, this is from harmony, which exports some nonsense columns
    Data = Data.dropna(axis=1, how='all', inplace=False)
    #Data = Data.drop(columns = Data.columns[Data.isnull().values.all(axis=0)].tolist())
    
    ### remove rows (cells) with any NaN values
    Data = Data.dropna(axis=0, how='any', inplace=False)
    
    ### removing some nonsense columns that cell profiler creates!
    nonesense_cols = ['Cells_Children_Cytoplasm_Count',
                                'Cyto_Number_Object_Number',
                                'Cyto_Parent_Cells',
                                'Cyto_Parent_Nuclei',
                                'Nuc_Children_Cytoplasm_Count',
                                'Nuc_Number_Object_Number',
                                'Nuc_Parent_NucleiIncludingEdges']
    
    for col in nonesense_cols:
        if Data.columns.isin([col]).any():
            Data = Data.drop(columns = [col])
        
    ### reset index
    Data = Data.reset_index(drop = True)
    
    ### end of loading data
    return Data


##### get feature vector
def get_feature_vector(df):
    ######
    ### get metadata columns and define feature columns after deleting "0" Features

    Feature_Ident = ["Nuc_", "Cells_", "Cyto_"]
    feat = []
    
    for col in df.columns.tolist():
        if any([col.startswith(x) for x in Feature_Ident]):
                feat.append(col)
    return feat


##### Regressing out Features

def Regressing_out(Data):
    start = time.time()
    Features = get_feature_vector(Data)

    x = Data["Cells_AreaShape_Area"].values
    x = x.reshape(-1, 1)

    for col in (set(Features)-set(["Cells_AreaShape_Area"])):
        y = Data[col].values

        reg = LinearRegression(fit_intercept = False)
        reg.fit(x, y)

        prediction = reg.predict(x)

        residuals = (y - prediction)

        Data[col] = residuals
        
    end = time.time()
    print("Time for regressing out cell area: ", round(((end-start)/60), 2), "min")

    return Data
    
    
##### HBOS: Histogram based outlier selection
def HBOS_selection(Data):
    start = time.time()
    Features = get_feature_vector(Data)

    HBOS_model = HBOS(n_bins=10, alpha=0.1, tol=0.5, contamination=0.1)
    HBOS_model.fit(Data.loc[:, Features].values)

    # get the prediction labels
    y_pred = HBOS_model.labels_  # binary labels (0: inliers, 1: outliers)
    
    ### remove cells
    Data["Metadata_Outlliers"] = y_pred

    Data = Data.loc[Data["Metadata_Outlliers"] == 0]
    Data = Data.reset_index(drop = True)

    end = time.time()

    print("Time for HBOS: ", round(((end-start)/60), 2), "min")
    print((y_pred == 1).sum(), "outlier cells removed")
    
    return Data



###
##### removing toxic conditions
###
### we will remove all wells where the cell count is < median - 2.5 std of the population

def remove_tox(Data, key_col = ["Metadata_EOS", "Metadata_Plate", "Metadata_Concentration"], SD_Threshold = 2.5,  plot_distribution = True):
    """
    removes toxic conditions from aggregated CellProfiler Profiles
    :param Data: aggregated CellProfiler Profiles
    :param key_col: list of column name used as key to identify treatments
    :param plot_distribution: boolean, if True histograms of cell count distribution will be plotted
    :return: new DataFrame without toxic conditions, set of toxic conditions
    """
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    Features = key_col.copy()
    Features.append("Metadata_Object_Count")

    Median_CellCount = pycytominer.consensus(
        profiles = Data.loc[:,Features], # A file or pandas DataFrame of profile data
        replicate_columns = key_col, # Metadata columns indicating which replicates to collapse, defaults to [“Metadata_Plate”, “Metadata_Well”]
        operation = "median", # (str) – The method used to form consensus profiles, defaults to “median”
        features = ["Metadata_Object_Count"], # (str, list) – The features to collapse, defaults to “infer”
    )

    tox_threshold = Median_CellCount["Metadata_Object_Count"].median() -(SD_Threshold*Median_CellCount["Metadata_Object_Count"].std())
    
    if tox_threshold < 50:
        tox_threshold = 50
    
    Tox_cond = Median_CellCount.loc[Median_CellCount["Metadata_Object_Count"] < tox_threshold, key_col]

    i1 = Data.set_index(key_col).index
    i2 = Tox_cond.set_index(key_col).index
    Data_tox = Data[~i1.isin(i2)]

    Data_tox = Data_tox.reset_index(drop = True)
    
    print("Toxic conditions removed with threshold", round(tox_threshold, 2))
    print("Old shape", Data.shape)
    print("New shape", Data_tox.shape)
    
    if plot_distribution == True:
        fig, axs = plt.subplots(ncols=2)

        sns.histplot(
            ax=axs[0],
            data = Data, 
            x = "Metadata_Object_Count",
            #hue= "Metadata_Plate"
        )
        axs[0].set_xlim([0, None])
        
        sns.histplot(
            ax=axs[1],
            data = Data_tox, 
            x = "Metadata_Object_Count",
            #hue= "Metadata_Plate"
        )
        axs[1].set_xlim([0, None])
        
        fig.suptitle('Cell Count')
        axs[0].set_title('Before')
        axs[1].set_title('After')
        
    
    return Data_tox, Tox_cond


###
##### Reproducibility
###

def corr_between_replicates_CP(df, group_by_feature):
    """
        Correlation between replicates
        Parameters:
        -----------
        df: pd.DataFrame
        group_by_feature: Feature name to group the data frame by
        Returns:
        --------
        list-like of correlation values
     """
    replicate_corr = []
    names = []
    replicate_grouped = df.groupby(group_by_feature)
    for name, group in replicate_grouped:
        group_features = group.loc[:, get_feature_vector(group)]
        corr = np.corrcoef(group_features)
        if len(group_features) == 1:  # If there is only one replicate on a plate
            replicate_corr.append(np.nan)
            names.append(name)
        else:
            np.fill_diagonal(corr, np.nan)
            replicate_corr.append(np.nanmedian(corr))  # median replicate correlation
            names.append(name)
    return replicate_corr, names
        

def corr_between_non_replicates_CP(df, n_samples, n_replicates, metadata_compound_name):
    """
        Null distribution between random "replicates".
        Parameters:
        ------------
        df: pandas.DataFrame
        n_samples: int
        n_replicates: int
        metadata_compound_name: Compound name feature
        Returns:
        --------
        list-like of correlation values, with a  length of `n_samples`
    """
    df.reset_index(drop=True, inplace=True)
    null_corr = []
    while len(null_corr) < n_samples:
        compounds = random.choices([_ for _ in range(len(df))], k = n_replicates)
        sample = df.loc[compounds].copy()
        if len(sample[metadata_compound_name].unique()) == n_replicates:
            sample_features = sample.loc[:, get_feature_vector(sample)]
            corr = np.corrcoef(sample_features)
            np.fill_diagonal(corr, np.nan)
            null_corr.append(np.nanmedian(corr))  # median non-replicate correlation
    return null_corr

def percent_score(null_dist, corr_dist, how):
    """
    Calculates the Percent strong or percent recall scores
    :param null_dist: Null distribution
    :param corr_dist: Correlation distribution
    :param how: "left", "right" or "both" for using the 5th percentile, 95th percentile or both thresholds
    :return: proportion of correlation distribution beyond the threshold
    """
    if how == 'right':
        perc_95 = np.nanpercentile(null_dist, 95)
        above_threshold = corr_dist > perc_95
        return np.mean(above_threshold.astype(float))*100, perc_95
    if how == 'left':
        perc_5 = np.nanpercentile(null_dist, 5)
        below_threshold = corr_dist < perc_5
        return np.mean(below_threshold.astype(float))*100, perc_5
    if how == 'both':
        perc_95 = np.nanpercentile(null_dist, 95)
        above_threshold = corr_dist > perc_95
        perc_5 = np.nanpercentile(null_dist, 5)
        below_threshold = corr_dist < perc_5
        return (np.mean(above_threshold.astype(float)) + np.mean(below_threshold.astype(float)))*100, perc_95, perc_5


##### Roproducibility score

def remove_non_reproducible(Data, n_samples = 100, n_replicates = 4, ID_col = "Metadata_Gene_ID", cntrls = ["DMSO", "Nocodazole", "Tetrandrine"], description = "Data"):

    corr_replicating_df = pd.DataFrame()

    replicating_corr, names = list(corr_between_replicates_CP(Data, ID_col))
    null_replicating = list(corr_between_non_replicates_CP(Data, n_samples=n_samples, n_replicates=n_replicates, metadata_compound_name = ID_col))

    prop_95_replicating, value_95_replicating = percent_score(null_replicating, replicating_corr, how='right')
    
    ### this only works well for big data sets with bigger 3 repetitions. Otherwise the null distribution is very wide! Thus we set a threshold for these cases
    
    if value_95_replicating > 0.6:
        value_95_replicating = np.float64(0.6)
        above_threshold = replicating_corr > value_95_replicating
        prop_95_replicating = np.mean(above_threshold.astype(float))*100


    corr_replicating_df = corr_replicating_df.append({'Description': description,
                                                       # 'Modality':f'{modality}',
                                                        #'Cell':f'{cell}',
                                                        #'time':f'{time}',
                                                        'Replicating':replicating_corr,
                                                        'Null_Replicating':null_replicating,
                                                        'Percent_Replicating':'%.1f'%prop_95_replicating,
                                                        'Value_95':value_95_replicating}, ignore_index=True)


    print(corr_replicating_df[['Description','Percent_Replicating']].to_markdown(index=False))
    
    ### remove non replicating conditions
    df = pd.DataFrame(list(zip(names, replicating_corr)),
               columns =['Name', 'replicating_corr'])
    
    replicating = list(df.loc[df["replicating_corr"] > value_95_replicating, "Name"])
    replicating.extend(cntrls)
    replicating = list(set(replicating))

    Data_replicating = Data.loc[Data[ID_col].isin(replicating)].copy()
    
    print("Nonreplicating conditions removed with threshold", round(value_95_replicating, 2))
    if len(set(names) - set(replicating)) == 0:
        print("No conditions below threshold")
    #else:
        #print(set(names) - set(replicating))
    print("Old shape", Data.shape)
    print("New shape", Data_replicating.shape)
    
    return Data_replicating, corr_replicating_df


###
##### Feature reduction
###

def feature_reduction(Data, variance_freq_cut=0.1, variance_unique_cut=0.1, outlier_cutoff=500, corr_threshold = 0.8, print_stats = True):
    """
    Reduces the number of featues by removing NaN columns, low variance, outlier and correlating features
    :param Data: DataFrame of well-averaged Profiles
    :param variance_freq_cut: frequencey cut off for the varience filter, defaults to 0.1
    :param variance_unique_cut: unique cut argument for the varience filter, defaults to 0.1
    :param outlier_cutoff: the threshold for the maximum or minimum value of a feature, defaults to 500
    :param corr_threshold: threshold for pearson correlation to determine correlating features, defaults to 0.8
    :param print_stats: boolean, if True prints statistics for each step
    :return: DataFarame with reduced features
    """

    ### 1) Remove NaN columns
    Data_DropNA = Data.drop(columns = Data.columns[(Data.isnull()).any()])

    ### 2) Remove Columns with low varience and thus low information

    Features = get_feature_vector(Data_DropNA)
        
    Data_VarianceThresh = pycytominer.feature_select(
            profiles = Data_DropNA, 
            features = Features, 
            samples='all', 
            operation='variance_threshold', 
            freq_cut=variance_freq_cut, # 2nd most common feature val / most common [default: 0.1]
            unique_cut=variance_unique_cut # float of ratio (num unique features / num samples) [default: 0.1] 
    )
    
    ### 3) Remove Columns with Outliers
    Features = get_feature_vector(Data_VarianceThresh)
     
    Data_dropOutliers = pycytominer.feature_select(
            profiles = Data_VarianceThresh, 
            features = Features, 
            samples='all', 
            operation='drop_outliers', 
            outlier_cutoff=outlier_cutoff # [default: 15] the threshold at which the maximum or minimum value of a feature
    )

    ### 4) Remove correlating features
    Features = get_feature_vector(Data_dropOutliers)
    Meta_Features = list(set(Data_dropOutliers.columns) - set(Features))
    
    Data_Reduced = pycytominer.feature_select(
            profiles = Data_dropOutliers, 
            features = Features, 
            samples='all', 
            operation='correlation_threshold',
            corr_threshold = corr_threshold, 
            corr_method='pearson'
    )
    
    if print_stats == True:
        Features_df = Feature_Vis(get_feature_vector(Data), "Original Features")
        Features_df = Features_df.merge(Feature_Vis(get_feature_vector(Data_VarianceThresh), "Variance Threshold"), on = "Category")
        Features_df["% Variance"] = round((100/Features_df["Original Features"])*Features_df["Variance Threshold"],1)
        Features_df = Features_df.merge(Feature_Vis(get_feature_vector(Data_dropOutliers), "Outlier Threshold"), on = "Category")
        Features_df["% Outlier"] = round((100/Features_df["Original Features"])*Features_df["Outlier Threshold"],1)
        Features_df = Features_df.merge(Feature_Vis(get_feature_vector(Data_Reduced), "Correlation Threshold"), on = "Category")
        Features_df["% Correlation"] = round((100/Features_df["Original Features"])*Features_df["Correlation Threshold"],1)
        print(Features_df.to_markdown(index=False))
        print()
        
        Features_df = Feature_Vis_Compartment(get_feature_vector(Data), "Original Features")
        Features_df = Features_df.merge(Feature_Vis_Compartment(get_feature_vector(Data_VarianceThresh), "Variance Threshold"), on = "Category")
        Features_df["% Variance"] = round((100/Features_df["Original Features"])*Features_df["Variance Threshold"],1)
        Features_df = Features_df.merge(Feature_Vis_Compartment(get_feature_vector(Data_dropOutliers), "Outlier Threshold"), on = "Category")
        Features_df["% Outlier"] = round((100/Features_df["Original Features"])*Features_df["Outlier Threshold"],1)
        Features_df = Features_df.merge(Feature_Vis_Compartment(get_feature_vector(Data_Reduced), "Correlation Threshold"), on = "Category")
        Features_df["% Correlation"] = round((100/Features_df["Original Features"])*Features_df["Correlation Threshold"],1)
        print(Features_df.to_markdown(index=False))
         
       
    return Data_Reduced

        
###
### Feature Reduction
### Which features are lost

def Feature_Vis(Feature_list, Ident):
    import re

    ## only intensity features
    r = re.compile(".*Intensity.*")
    Intensity_Features = list(filter(r.match, Feature_list))

    ## only Correlation features
    r = re.compile("Nuc_Correlation|Cells_Correlation|Cyto_Correlation.*")
    Correlation_Features = list(filter(r.match, Feature_list))

    ## only AreaShape features
    r = re.compile(".*AreaShape.*")
    AreaShape_Features = list(filter(r.match, Feature_list))

    ## only Granularity features
    r = re.compile(".*Granularity.*")
    Granularity_Features = list(filter(r.match, Feature_list))

    ## only Neighbors features
    r = re.compile(".*Neighbors.*")
    Neighbors_Features = list(filter(r.match, Feature_list))

    ## only RadialDistribution features
    r = re.compile(".*RadialDistribution.*")
    RadialDistribution_Features = list(filter(r.match, Feature_list))

    ## only Texture features
    r = re.compile(".*Texture.*")
    Texture_Features = list(filter(r.match, Feature_list))

    ## mito skeleton
    r = re.compile(".*ObjectSkeleton.*")
    Skeleton_Features = list(filter(r.match, Feature_list))

    Found_features = []
    Found_features.extend(Intensity_Features)
    Found_features.extend(Correlation_Features)
    Found_features.extend(AreaShape_Features)
    Found_features.extend(Granularity_Features)
    Found_features.extend(Neighbors_Features)
    Found_features.extend(RadialDistribution_Features)
    Found_features.extend(Texture_Features)
    Found_features.extend(Skeleton_Features)

    
    df_features = pd.DataFrame()
    df_features["Category"] = ["Total Features",
                               "Intensity",
                               "Correlation",
                               "AreaShape",
                               "Granularity",
                               "Neighbors",
                               "RadialDistribution",
                               "Texture",
                               'MitoSkeleton'] 
    df_features[Ident] = [len(Feature_list),
                          len(Intensity_Features),
                          len(Correlation_Features),
                          len(AreaShape_Features),
                          len(Granularity_Features),
                          len(Neighbors_Features),
                          len(RadialDistribution_Features),
                          len(Texture_Features),
                          len(Skeleton_Features)]
   
    return df_features


### Feature visualization by compartment
def Feature_Vis_Compartment(Feature_list, Ident):
    import re

    ## only nuclear features
    r = re.compile(".*Nuc.*")
    Nucleus_Features = list(filter(r.match, Feature_list))

    ## only Cell features
    r = re.compile(".*Cell.*")
    Cell_Features = list(filter(r.match, Feature_list))

    ## only Cytoplasm features
    r = re.compile(".*Cyto.*")
    Cyto_Features = list(filter(r.match, Feature_list))

    
    Found_features = []
    Found_features.extend(Nucleus_Features)
    Found_features.extend(Cell_Features)
    Found_features.extend(Cyto_Features)
    
    df_features = pd.DataFrame()
    df_features["Category"] = ["Total Features",
                               "Nucleus",
                               "Cell",
                               "Cytoplasm"] 
    df_features[Ident] = [len(Feature_list),
                          len(Nucleus_Features),
                          len(Cell_Features),
                          len(Cyto_Features)]
   
    return df_features 


###
##### Visualizations
###
##### UMAP

def UMAP_proj(Data, dim = 2):
    """
    UMAP projection of profiles
    :param Data: DataFrame of Profiles
    :param dim: Dimensions of the UMAP dimensionality reduction
    :return: DataFarame UMAP axis and original Metadata
    """
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP
    
    Data = Data.reset_index(drop = True)
    Features = get_feature_vector(Data)
    Meta_Features = list(set(Data.columns) - set(Features))
    
    # Normalization
    x = Data.loc[:, Features]
    x = StandardScaler().fit_transform(x) # normalizing the features. Actually the data already is normalized by pyCytoMiner!!!!!
    x.shape

    #Let's convert the normalized features into a tabular format with the help of DataFrame.
    feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
    normalised_Features = pd.DataFrame(x,columns=feat_cols)

    umap = UMAP(n_components=dim, init='random', random_state=0)

    proj = umap.fit_transform(normalised_Features)

    ## Next, let's create a DataFrame that will have the principal component values
    if dim == 2:
        proj_df = pd.DataFrame(data = proj,
                               columns = ['Axis 1', 'Axis 2'])
        
    if dim == 3:
        proj_df = pd.DataFrame(data = proj,
                               columns = ['Axis 1', 'Axis 2', 'Axis 3'])
        
        
    UMAP_Data = pd.merge(Data.loc[:, Meta_Features], proj_df, left_index = True, right_index = True)
    print("UMAP projection performed")
    
    return UMAP_Data

###
##### kmeans clustering with sorting of centroids
###

def clustering(Data, n_cluster, neg_cntrl = 'Metadata_EOS=="DMSO"', saveFig = "./Figures/MST.png", UMAP = False):
    """
    kMeans clustering and centroid sorting
    :param Data: DataFrame of Profiles
    :param n_cluster: Number of Clusters
    :param neg_cntrl: negative control identification in for of query: 'Metadata_EOS=="DMSO"'
    :param saveFig: path for saving MST plot
    :param UMAP: boolean, if True UMAP data expected, Features = ["Axis 1", "Axis 2", "Axis 3"]
    :return: DataFrame with kmeans clusters, kmeans model
    """
    ### get Features
    if UMAP == True:
        Features = ["Axis 1", "Axis 2", "Axis 3"]
    else:
        Features = get_feature_vector(Data)

    Meta_Features = set(Data.columns) - set(Features)
    
    ### normalize data
    X = StandardScaler().fit_transform(Data.loc[:, Features])

    normalised_Features = pd.DataFrame(X, columns=Features)
    Data_Norm = pd.merge(Data.loc[:, Meta_Features], normalised_Features, left_index = True, right_index = True)
    
    ### kmeans clustering
    model = KMeans(n_clusters = n_cluster, random_state=0, n_init = 100 )
    model.fit(X)

    ### sort centroids to distance from DMSO
    df_negcon = Data_Norm.query(neg_cntrl)
    negcon_clusters = model.predict( df_negcon[Features].values)
    negcon_cluster = stats.mode( negcon_clusters )[0][0]
    model = sort_centroids( model, negcon_cluster , metric='euclidean')

    ### predict cluster of data
    Data['Cluster'] = model.predict(X)

    ### Create centroids table
    centroids_df = pd.DataFrame(model.cluster_centers_, columns=Features)
    centroids_df['Cluster'] = range(0,model.cluster_centers_.shape[0])
    centroids_df = centroids_df[ ['Cluster'] + Features] 

    ### Create MST 
    mst = create_MST( model.cluster_centers_, 'euclidean')

    # Define display node size
    node_size = Data['Cluster'].value_counts().reset_index()
    node_size.rename(columns={'Cluster': 'profile_count', 'index': 'Cluster'}, inplace=True)
    node_size['size'] = round(np.log2(node_size.profile_count)+1).astype(int) * 75

    # defining node colors
    node_colors = plt.cm.jet(np.linspace(0,1,len(mst.nodes)))    
    gray = np.array([200/256,200/256,200/256, 1])

    negcon_clusters = model.predict( df_negcon[Features].values)
    negcon_cluster = stats.mode( negcon_clusters )[0][0] 
    for i in set(negcon_clusters):
        node_colors[i, :] = gray
        
    non_DMSO_clusters = list(set(Data['Cluster']).difference(set(negcon_clusters)))
    colors = plt.cm.jet(np.linspace(0,1,len(non_DMSO_clusters)))    
    for j, k in enumerate(non_DMSO_clusters): 
        node_colors[k, :] = colors[j, :]

    ### plot MST
    draw_numbers = True
    plot_node_size = node_size.sort_values(by=['Cluster'])['size']

    fig, ax = plt.subplots(num=None, figsize=(10,10), facecolor='w', edgecolor='k')
    labels = dict()
    for c in range(len(mst.nodes)): labels[c] = str(c)
    nodes_pos = nx.kamada_kawai_layout(mst)  # positions for all nodes
    #node_colors = plt.cm.jet(np.linspace(0,1,len(mst.nodes)))    
    nx.draw_networkx_nodes( mst, nodes_pos, nodelist=mst.nodes, node_color=node_colors, alpha=1, node_size=plot_node_size)
    nx.draw_networkx_edges( mst, nodes_pos, edgelist=mst.edges, edge_color='k', alpha=0.5)
    if draw_numbers:
        text = nx.draw_networkx_labels( mst, nodes_pos, labels=labels, font_size=12)
        for _,t in text.items():
            t.set_rotation(0)
    plt.xticks([])
    plt.yticks([])
    #plt.show()
    plt.savefig(saveFig, transparent=False, bbox_inches='tight', dpi = 600)

    return Data, model


###
##### sort centroids
###
def sort_centroids( model, ref_centroid , metric="euclidean"):
    """
    Exchanges centroids such that they are sorted by increasing distance to ref centroid"
    """
    centroids = model.cluster_centers_    
    centroid_distances = distance.cdist( centroids , centroids , metric=metric)
    D_ref = centroid_distances[ref_centroid,:] # distance relative to the ref cluster
    sorted_idx = np.argsort(D_ref)
    temp = centroids.copy()
    for i, idx in enumerate(sorted_idx):
        centroids[i,:]= temp[idx,:]
    model.cluster_centers_ = centroids 

    return model


###
##### make MST
###
def create_MST(centroids, metric):
    """ 
    Creates minumum spanning tree from cluster centroids, i.e, a connected graph whose sum of edge weights is as small as possible.  
    """
    #Nclusters = centroids.shape[0]
    centroid_distances = distance.cdist( centroids , centroids , metric=metric )
    G   = nx.Graph(centroid_distances)
    mst = nx.minimum_spanning_tree(G) 
    
    return mst



###
##### Plotting UMAP projections with cluster info
###

def plot_UMAP_Cluster(Data, col, dim, filename, plot_title):
    
    ### defining node colors for reduced node number
    node_colors = plt.cm.jet(np.linspace(0,1,len(set(Data[col]))-1 ))    
    gray = np.array([200/256,200/256,200/256, 1])
    node_colors = np.append([gray], node_colors, axis=0)

    Hex_col = []
    for i in range(0,len(node_colors)):
        Hex_col.append(mpl.colors.to_hex(node_colors[i], keep_alpha=False))

    ### plot 2D
    if dim == 2:
        ### using the node colors of the MST
        colors = sns.color_palette(node_colors)

        p1 = sns.relplot(
            data = Data,
            x="Axis 1", y="Axis 2",
            hue = col,
            palette = colors,
            #legend = None,
            s = 20,
            #style = "Concentration"
            height=10, aspect=10/10
        )

        p1.set(xlabel = "UMAP Axis 1", 
            ylabel = "UMAP Axis 2", 
            title = plot_title)
        
        p1.figure.savefig(filename, transparent=False, bbox_inches='tight', dpi = 600)

    if dim == 3:
        Data_plot = Data.copy()

        Data_plot = Data_plot.fillna("Not Defined")
        Data_plot["Cluster"] = Data_plot[col].astype("str")
        Data_plot = Data_plot.sort_values(by=[col])

        ### plot

        fig = px.scatter_3d(Data_plot, x='Axis 1', y='Axis 2', z='Axis 3',
                            color_discrete_sequence = Hex_col,
                            color = "Cluster",
                            hover_data=['Metadata_EOS', 'Metadata_Object_Count'])

        fig.update_layout(
            autosize=False,
            width=1500,
            height=900,
            #title = "Well: " + Selected_Compounds.loc[Images_to_Display, "Well"] + "Cluster: " + Selected_Compounds.loc[Images_to_Display, "Cluster"])
            )

        ### marker size
        fig.update_traces(marker={'size': 4})
        fig.write_html(filename)

###
##### Cell Count histogram per cluster
###
def count_hist(Data, col, filename):
    ### defining node colors for reduced node number
    node_colors = plt.cm.jet(np.linspace(0,1,len(set(Data[col]))-1 ))    
    gray = np.array([200/256,200/256,200/256, 1])
    node_colors = np.append([gray], node_colors, axis=0)

    Hex_col = []
    for i in range(0,len(node_colors)):
        Hex_col.append(mpl.colors.to_hex(node_colors[i], keep_alpha=False))

    ax = sns.displot(
        Data, 
        x = "Metadata_Object_Count",
        hue= col,
        #col = "Treatment",
        kind="kde",
        palette = Hex_col
    )
    ax.set(xlim=(0, None))
    ax.figure.savefig(filename, transparent=False, bbox_inches='tight', dpi = 600)

###
##### Clustergrammer data prep
### metadata columns

def clustergrammer_meta_col(Features):
    import re

    Data_meta = pd.DataFrame(Features)
    Data_meta = Data_meta.set_index(0)
    Data_meta.index.name = None

    # Nucleus
    subs = 'Nuc_'
    # using list comprehension 
    # to get string with substring 
    Nuc = [i for i in Features if subs in i]

    # Cytoplasm
    subs = 'Cyto'
    # using list comprehension 
    # to get string with substring 
    Cyto = [i for i in Features if subs in i]

    # Cell
    subs = 'Cell'
    # using list comprehension 
    # to get string with substring 
    Cell = [i for i in Features if subs in i]

    Data_meta.loc[Nuc, "Compartment"] = "Nucleus"
    Data_meta.loc[Cyto, "Compartment"] = "Cytoplasm"
    Data_meta.loc[Cell, "Compartment"] = "Cell"

    ## only intensity features
    r = re.compile(".*Intensity.*")
    Intensity_Features = list(filter(r.match, Features))

    ## only Correlation features
    r = re.compile("Nuc_Correlation|Cells_Correlation|Cyto_Correlation.*")
    Correlation_Features = list(filter(r.match, Features))

    ## only AreaShape features
    r = re.compile(".*AreaShape.*")
    AreaShape_Features = list(filter(r.match, Features))

    ## only Granularity features
    r = re.compile(".*Granularity.*")
    Granularity_Features = list(filter(r.match, Features))

    ## only Neighbors features
    r = re.compile(".*Neighbors.*")
    Neighbors_Features = list(filter(r.match, Features))

    ## only RadialDistribution features
    r = re.compile(".*RadialDistribution.*")
    RadialDistribution_Features = list(filter(r.match, Features))

    ## only Texture features
    r = re.compile(".*Texture.*")
    Texture_Features = list(filter(r.match, Features))

    Data_meta.loc[Intensity_Features, "Measurement"] = "Intensity"
    Data_meta.loc[Correlation_Features, "Measurement"] = "Correlation"
    Data_meta.loc[AreaShape_Features, "Measurement"] = "AreaShape"
    Data_meta.loc[Granularity_Features, "Measurement"] = "Granularity"
    Data_meta.loc[Neighbors_Features, "Measurement"] = "Neighbors"
    Data_meta.loc[RadialDistribution_Features, "Measurement"] = "RadialDistribution"
    Data_meta.loc[Texture_Features, "Measurement"] = "Texture"


    ### channel
    ## DNA
    r = re.compile(".*DNA.*")
    DNA_Features = list(filter(r.match, Features))

    ## AGP
    r = re.compile(".*AGP.*")
    AGP_Features = list(filter(r.match, Features))

    ## ER
    r = re.compile(".*ER.*")
    ER_Features = list(filter(r.match, Features))

    ## Mito
    r = re.compile(".*Mito.*", re.IGNORECASE)
    Mito_Features = list(filter(r.match, Features))


    Data_meta.loc[DNA_Features, "Channel"] = "DNA"
    Data_meta.loc[AGP_Features, "Channel"] = "AGP"
    Data_meta.loc[ER_Features, "Channel"] = "ER"
    Data_meta.loc[Mito_Features, "Channel"] = "Mito"

    return Data_meta



###
##### Biosimilarity for FMP comps
###
def BioSim_Table(data, Comp_Name, BioSimThreshold = 0.75, Exclude_FMP = True):
    from scipy.spatial.distance import correlation
    Features = get_feature_vector(data)
    Data_Temp = []

    Concentrations = list(set(data.loc[(data["Metadata_EOS"] == Comp_Name), "Metadata_Concentration"]))
    Concentrations.sort()

    for Comp_Conc in Concentrations: 
        
        ### generate Compound profile to compare to
        Comp_Profile = data.loc[(data["Metadata_EOS"] == Comp_Name) & (data["Metadata_Concentration"] == Comp_Conc), Features].values
        ### exclude the searched Profile
        if Exclude_FMP == True:
            df_BioActives = data.loc[~((data["Metadata_EOS"] == Comp_Name) & (data["Metadata_Concentration"] == Comp_Conc))].copy()
            df_BioActives = df_BioActives.loc[df_BioActives["Metadata_Partner"] == "Bioactives"]
        else:
            df_BioActives = data.loc[~((data["Metadata_EOS"] == Comp_Name) & (data["Metadata_Concentration"] == Comp_Conc))].copy()
        
        ### calculate distance correlation between Comp_profile and all BioActives
        dist_corr = []

        for i in df_BioActives.index:
            Y = df_BioActives.loc[i, Features].values
            corr_temp = correlation(Comp_Profile, Y, centered = True)
            dist_corr.append(corr_temp)

        ### Calculate BioSimilarity    
        BioSim = 1-np.array(dist_corr)

        ### create Nice DataFrame
        Meta_Features = list(set(df_BioActives.columns)-set(Features))
                
        Data_BioSim = df_BioActives.loc[:, Meta_Features].copy()
        Data_BioSim["BioSim"] = BioSim

        ### Filter for only the compounds with high similarity
        Data = Data_BioSim.loc[Data_BioSim["BioSim"] > BioSimThreshold].sort_values(by="BioSim", key=abs, ascending=False)
        Data = Data.reset_index(drop=True)

        Data["Compounds_Similar_to"] = Comp_Name+" "+str(Comp_Conc)+" uM"
        cols = ["Compounds_Similar_to", "Metadata_EOS", "Metadata_Concentration", "EUopen_name", "Metadata_Object_Count", "BioSim", "Gene_Target_Names", "EUopen_target_name", "Broad_MOA","JUMP-CP_moa", "JUMP-Target_target", "MESH_MeSH_Class"]

        df = Data.loc[:, cols].copy()
        if len(df):
            Data_Temp.append(df)

    if len(Data_Temp) > 0:
        Data_All = pd.concat(Data_Temp)
        ### save
        filename = "./Files_BioSimilarity/" + str(date.today()) +"_"+ Comp_Name + "_List of Biosimilar compounds > " + str(BioSimThreshold) + ".csv"
        Data_All.to_csv(filename, index = False)
        
        print(Comp_Name, "has Biosimilar compounds at Threshold", BioSimThreshold)
        print("")

    else:
        print(Comp_Name, "has no compounds with BioSimilarity >", BioSimThreshold, "and is not close to any other compounds")
        print("")

###
##### Feature Importance plot
###
def plot_feature_importance(df, filename, cluster_plot = False):
    Features = get_feature_vector(df)
    ### median cluster response
    median_ovelap = df.loc[:, Features].median()
    sorted_median_overlap = median_ovelap.sort_values(axis = 0, key=abs, ascending=False)
    best_Feat = list(sorted_median_overlap.loc[abs(sorted_median_overlap) > 2].index)

        
    ### if best_Feat gets too long we stop at 50
    if len(best_Feat) > 50:
        best_Feat = best_Feat[:49]

    ### stop if no feature is above 2
    if len(best_Feat) == 0:
        print("no Features change significantly")
        return

    #best_Feat = ['Cyto_Intensity_IntegratedIntensity_AGP', 'Cells_Correlation_K_DNA_AGP']
    ### get transformed data to plot
    data = df.loc[:, best_Feat]

    ### drop columns where there are crazy values
    data = data.loc[:, (abs(data).max(axis=0) < 1000)]

    idx = list(data.index)
    trans = data.transpose()
    trans = trans.reset_index()
    trans.rename(columns={"index": "Features"}, inplace=True)
    for i, ind in enumerate(idx):
        name = "Rep"+str(i+1)
        trans.rename(columns={ind: name}, inplace=True)

    ### making wide format out of it
    wide = pd.wide_to_long(trans, ['Rep'], i='Features', j='Repetition')
    wide = wide.reset_index()
    wide.rename(columns={"Rep": "Z-Score"}, inplace=True)

    ### make boxplot with all high features
    if cluster_plot == True:
        plot_title = "Feature Impact for cluster " + str(set(df.Cluster_15))
    else:
        plot_title = "Feature Impact for" + str(set(df.Metadata_EOS)) + "at" + str(set(df.Metadata_Concentration)) + "uM"
    plt.figure(figsize = ((len(best_Feat)/5),8))
    ax = sns.boxplot(
        data = wide, 
        x = "Features",
        y = "Z-Score"
    )

    #loc, labels = plt.xticks()
    #ax.set_xticklabels(labels, rotation=90)
    ax.axhline(0)
    ax.set_xticklabels(ax.get_xticklabels(), 
                                rotation=90, 
                                horizontalalignment='center')

    ax.set(xlabel = "", 
            title = plot_title)
    #plt.show()

    plt.savefig(filename, transparent=False, bbox_inches='tight', dpi = 600)
    plt.close()

        