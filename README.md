# imoco4e-pilot3-etfa2023
Repository of the experimental setup for the paper "Vector Reconstruction Error for Anomaly Detection: Preliminary Results in the IMOCO4.E Project"
It should be noted that the following packages are needed to run our experimental setup:
- pyNeVer and all its required packages
- sklearn
- pandas
- matplotlib

Furthermore, given the size of the data, the user needs to autonomously download the
[dataset](https://www.kaggle.com/datasets/inIT-OWL/one-year-industrial-component-degradation) and paste the folder
"oneyeardata/" in the folder *data/*.

The results, models and graph presented in the paper can be found in the folder *exp_paper_res/*.
If the user want to re-generate the models, the following scripts must be launched in order:
- *data_preprocessing.py*
- *models_generation.py*
- *loss_analysis.py*

The results obtained by this process may slightly differ from the one presented in the paper
due to the pseudo-randomic nature of some algorithms used.