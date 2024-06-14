Project :Auto Ml
Introduction
Welcome ! This project aims to tackle a specific problem using data science and machine learning techniques. In this document, you'll find detailed instructions on setting up the project environment, data preparation, model training, evaluation, and submission for a Kaggle competition
Installation
To get started with the project, you need to install the required dependencies. These dependencies are listed in the requirement.txt file. You can install them using the following commands:
!pip install -f requirement.txt #autogluon pandas openfe
!pip install kaggle
!pip install python-dotenv
Kaggle Configuration
Before proceeding, you'll need to set up Kaggle authentication. Follow these steps:

Obtain your Kaggle API token from your Kaggle account settings.
Save the token as kaggle.json.
Execute the following code to set up Kaggle authentication:
import os
os.environ["secret"] = "0522d5d835031612d2bda57db662295f"

!mkdir /home/jupyter/.kaggle
!cp kaggle.json /home/jupyter/.kaggle/kaggle.json
!sudo chmod 600 /home/jupyter/.kaggle/kaggle.json
!cat /home/jupyter/.kaggle/kaggle.json
Data Preparation
To prepare the data for analysis, follow these steps:
Load the training and test datasets using Pandas and Autogluon.
Clean and preprocess the datasets as needed.
import pandas as pd
from autogluon.tabular import TabularDataset

data_url_train = '~/imported/kaggle/input/playground-series-s4e5/train.csv'
train_data = TabularDataset(data_url_train)

data_url_test = '~/imported/kaggle/input/playground-series-s4e5/test.csv'
test_data = TabularDataset(data_url_test)
Feature Engineering
Feature engineering is a crucial step in the data preparation process. Here's an example of feature engineering techniques applied to the dataset:
def features(df):
    df_ = df.copy()
    cols = ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
            'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
            'Siltation', 'AgriculturalPractices', 'Encroachments',
            'IneffectiveDisasterPreparedness', 'DrainageSystems',
            'CoastalVulnerability', 'Landslides', 'Watersheds',
            'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
            'InadequatePlanning', 'PoliticalFactors']
    
    df_["sum"] = df[cols].sum(axis=1)
    df_["mean"] = df[cols].mean(axis=1)
    df_["std"] = df[cols].std(axis=1)
    df_["ptp"] = np.ptp(df[cols],axis=1)
    df_["norm"] = np.linalg.norm(df[cols],axis=1)
    df_['special'] = df_['sum'].isin(np.arange(72, 76)).astype(str)
    
    # Total Infrastructure Score
    df_['InfrastructureScore'] = df_['DeterioratingInfrastructure'] + df_['DrainageSystems'] + df_['DamsQuality']
    
    # Environmental Impact Score
    df_['EnvironmentalScore'] = df_['Deforestation'] + df_['Urbanization'] + df_['AgriculturalPractices'] + df_['WetlandLoss']
    
    # Socio-Political Score
    df_['SocioPoliticalScore'] = df_['PoliticalFactors'] + df_['IneffectiveDisasterPreparedness'] + df_['InadequatePlanning']
    
    # Geographical Vulnerability Score
    df_['GeographicalVulnerabilityScore'] = df_['TopographyDrainage'] + df_['CoastalVulnerability'] + df_['Landslides'] + df_['Watersheds']
    
    # Cumulative Human Activity Score
    df_['HumanActivityScore'] = df_['PopulationScore'] + df_['Encroachments'] + df_['AgriculturalPractices']
    
    # Interaction Terms
    df_['MonsoonIntensity_RiverManagement'] = df_['MonsoonIntensity'] * df_['RiverManagement']
    df_['Deforestation_Urbanization'] = df_['Deforestation'] * df_['Urbanization']
    
    # Temporal Features (if applicable)
    # Example: Month of the year
    # df_['Month'] = pd.to_datetime(df_['Date']).dt.month
    
    # Composite Indices
    df_['InfrastructureEnvironmentalIndex'] = (df_['InfrastructureScore'] + df_['EnvironmentalScore']) / 2
    df_['SocioGeographicalIndex'] = (df_['SocioPoliticalScore'] + df_['GeographicalVulnerabilityScore']) / 2
    
    return df_

train_data = features(train_data)
test_data = features(test_data)
Model Training
With the features engineered, we can proceed to train our machine learning model using Autogluon and OpenFE. Here's a snippet to demonstrate the training process:
from openfe import OpenFE
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

n_jobs = 32
ofe = OpenFE()
features = ofe.fit(data=train_data.drop(columns=[label,"id"]), label=train_data[label], n_jobs=n_jobs,
                   n_data_blocks=8, 
                   task='regression',
                   verbose = False,
                   metric="rmse"
                  )  # generate new features

train_x, test_x = transform(train_data.drop(columns=[label,"id"]), test_data.drop(columns=["id"]), ofe.new_features_list[:20], n_jobs=n_jobs) # transform the train and test data according to generated features.

train_0 = pd.concat([train_x,train_data[[label]]],axis=1)
train_data = train_0.copy()
test_data = test_x.copy()

train_data.to_parquet("train_feat.parquet",index=False)
test_data.to_parquet("test_feat.parquet",index=False)
Model Evaluation
X_train, X_test = train_test_split(
     train_data , test_size=0.01, random_state=42,stratify=train_data.FloodProbability)

Presets =['best_quality’,‘high_quality’,‘good_quality’,‘medium_quality’,‘optimize_for_deployment’,‘interpretable’,‘ignore_text’]
metrics = ['r2']

predictor = TabularPredictor(label=label,eval_metric=metrics[0],problem_type="regression").fit(X_train,
                                                                                               time_limit=int(3.5*60*60),
                                                                                               presets="best_quality",
                                                                                               # ag_args_fit={'num_cpus': 32}
                                                                                              )
predictor.evaluate(X_test)

LB = predictor.leaderboard(X_test)
test_preds = predictor.predict(test_data)
Submission
Finally, we generate predictions for the test dataset and submit them to the Kaggle competition:
submission.FloodProbability = test_preds.values
submission.to_csv("submission.csv",index=False)
Results were:
	model	score_test	score_val	eval_metric	pred_time_test	pred_time_val	fit_time	pred_time_test_marginal	pred_time_val_marginal	fit_time_marginal	stack_level	can_infer	fit_order
0	WeightedEnsemble_L2	0.868041	0.868686	r2	0.860950	90.327706	1165.777763	0.002203	0.008882	1.550873	2	True	14
1	NeuralNetFastAI_BAG_L2	0.868029	0.868480	r2	9.204198	957.696607	5572.368386	0.535035	7.170238	341.637860	2	True	20
2	NeuralNetFastAI_r191_BAG_L2	0.867975	0.867853	r2	10.185770	962.907171	6585.617518	1.516608	12.380803	1354.886992	2	True	26
3	CatBoost_BAG_L1	0.867919	0.868585	r2	0.134225	0.807645	612.355350	0.134225	0.807645	612.355350	1	True	4
4	WeightedEnsemble_L3	0.867891	0.868819	r2	9.273643	1024.180767	5746.697878	0.002502	0.009013	2.589254	3	True	27
5	RandomForestMSE_BAG_L1	0.867880	0.868025	r2	0.225554	48.234764	210.760798	0.225554	48.234764	210.760798	1	True	3
6	CatBoost_r177_BAG_L1	0.867870	0.868553	r2	0.107668	0.460493	196.756900	0.107668	0.460493	196.756900	1	True	10
7	CatBoost_BAG_L2	0.867844	0.868594	r2	8.761676	950.851656	5332.328446	0.092514	0.325287	101.597920	2	True	18
8	CatBoost_r177_BAG_L2	0.867805	0.868565	r2	8.753717	951.188780	5302.217196	0.084554	0.662411	71.486670	2	True	23
9	XGBoost_BAG_L2	0.867784	0.868716	r2	8.836808	954.333819	5272.430182	0.167645	3.807450	41.699656	2	True	21
10	XGBoost_BAG_L1	0.867641	0.868425	r2	0.192022	3.868304	41.814316	0.192022	3.868304	41.814316	1	True	7
11	ExtraTreesMSE_BAG_L1	0.867597	0.868040	r2	0.199276	36.947618	102.539526	0.199276	36.947618	102.539526	1	True	5
12	RandomForestMSE_BAG_L2	0.867516	0.868181	r2	8.897949	985.219684	5561.324834	0.228786	34.693316	330.594308	2	True	17
13	ExtraTreesMSE_BAG_L2	0.867485	0.868557	r2	8.874710	985.670988	5371.814660	0.205547	35.144619	141.084135	2	True	19
14	NeuralNetFastAI_BAG_L1	0.867211	0.867762	r2	1.038598	13.261893	2289.865908	1.038598	13.261893	2289.865908	1	True	6
15	NeuralNetFastAI_r191_BAG_L1	0.864842	0.862753	r2	0.873098	81.127638	299.045042	0.873098	81.127638	299.045042	1	True	13
16	NeuralNetTorch_r79_BAG_L2	0.861777	0.861754	r2	9.032217	955.240449	6045.735280	0.363055	4.714080	815.004755	2	True	24
17	NeuralNetTorch_r79_BAG_L1	0.859386	0.859306	r2	0.358131	136.891581	845.461812	0.358131	136.891581	845.461812	1	True	11
18	NeuralNetTorch_BAG_L1	0.850821	0.850704	r2	0.268776	5.610372	609.446697	0.268776	5.610372	609.446697	1	True	8
19	KNeighborsUnif_BAG_L1	0.832886	0.833703	r2	2.696687	313.979181	1.638050	2.696687	313.979181	1.638050	1	True	1
20	KNeighborsDist_BAG_L1	0.832870	0.833699	r2	2.472536	308.324506	1.784874	2.472536	308.324506	1.784874	1	True	2
21	LightGBMXT_BAG_L2	0.821052	0.807152	r2	8.706754	951.017630	5240.876893	0.037591	0.491261	10.146368	2	True	15
22	LightGBMLarge_BAG_L2	0.667237	0.661955	r2	8.724063	951.094753	5256.613849	0.054900	0.568384	25.883323	2	True	22
23	LightGBM_BAG_L2	0.633381	0.628195	r2	8.702097	950.916243	5239.164393	0.032935	0.389874	8.433867	2	True	16
24	LightGBM_r131_BAG_L2	0.624473	0.620803	r2	8.745659	951.207714	5241.797354	0.076497	0.681345	11.066828	2	True	25
25	LightGBMLarge_BAG_L1	0.596793	0.595238	r2	0.052856	0.470775	9.308696	0.052856	0.470775	9.308696	1	True	9
26	LightGBM_r131_BAG_L1	0.588842	0.587250	r2	0.049734	0.541600	9.952558	0.049734	0.541600	9.952558	1	True	12