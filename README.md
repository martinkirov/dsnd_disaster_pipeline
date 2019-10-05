# Disaster Response Pipeline Project

### Summary
The following repository contains all the code for the Disaster Response assignment to Udacity's Data Science for Enterprise Nanodegree. The assignment consists of completing three sections: an ETL pipeline to process a sample dataset, an ML pipeline to learn on the data and a Flask webapp to visualise some results and to enable on-the-fly predictions.

### Repository files explanation
- data contains all the files for the ETL pipeline: the initial csv input files, the resulting sqlite database and the ETL python script to process the data
- models contains the ML pipeline that takes the processed data from the sqlite database and produces a pickle instance of the machine learning model. (due to GitHub's size limitations, run the code in the instructions to get the .pkl file)
- app contains all the Flask files to run the webapp. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
