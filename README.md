Dataset:
https://github.com/yahoojapan/JGLUE?tab=readme-ov-file

Files:
-.env.example is where any API keys are contained
-environment.yml is used to create the environment for the project using conda
-analyze_dataset_complexity.py is a python script used to do an analysis over the rewritten dataset, outputting the results for each Keigo's avg length, avg jaccard, func/cont count, and the top POS for each of the Keigo
-evaluator.py is a python script that evaluates the LLM performance over the rewritten dataset, it goes through the dataset for every Keigo and for the original question recording the accuracy for each, outputting a json file for each Keigo accuracy
-inspect_data.py is a python script used to inspect the dataset use to understand how it should be rewritten
-rewriter.py is a python script used to rewrite the dataset so that it includes all four Keigo versions of the original dataset, it then saves this into a json to be used for evaluation
-rewriter_bar.py is a python script that would be used for future work to rewrite a Japanese Bar Exam dataset, but we have yet to gain access to this dataset to do tests
-test_api_access.py is a python script that is used to test the API key and make sure that it is usable for the project
-data folder contains all json data files of accuracy and the rewritten dataset
