# Disaster Response Pipeline Project
### Table of Contents

1. [Installation](#installation)
2. [Run Instructions](#run_instructios)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

All the required libraries are avaibale in the Udacity workspace environemnt.
The latest Anaconda2 2022.10 Python 3.9+ are sufficient in a local environment(PyCharm) taking into account this bug note:
 
Bug Note: Version 2.0.0 (January 26, 2023) of SQLAlchemy is not compatible with earlier versions of pandas.
-> latest pandas version 1.5.3 is required
 Reference: https://stackoverflow.com/questions/75282511/df-to-table-throw-error-typeerror-init-got-multiple-values-for-argument


## Run Instructions <a name="run_instructios"></a>
1. Run the following commands in the project's root directory to set up the database and model.
Run the ETL pipeline that cleans data and stores in database
   
`python /data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/<DisasterResponse.db> <Messages_Categories>`
2. Run the ML pipeline that trains classifier and saves the model 
   
`python models/train_classifier.py /data/<DisasterResponse.db> <Messages_Categories> models/<model.pkl>`

3. From `app` directory run the web app:
   
`python run.py`

4. Click the `PREVIEW` button to open the homepage to view the visualization part 

## Project Motivation<a name="motivation"></a>

For this project, I was interested in practicing the foundations of the end-to-end data science process flow including data engineering,
machine learning and visualization of results.

## File Descriptions <a name="files"></a>

There are 2 notebooks available as part of preparation for the executanle ETL (process_data.py) end ML (train_classifier.py) pipelines in the Notebooks folder

The ETL pipeline process_data.py as well as the source dsata files and the cleaned data base are in the Data folder.

The ML pipeline ML train_classifier.py and the ML model.pkl are in the Models folder.

The Flask application and HTNL templates are in the App folder


## Results<a name="results"></a>

The main outcome of the project is the ML model model.pkl

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

No licensing is required to leverage any part of this repository.
A big credit to:
1. Udacity coursework "Data Scientist" Section which code samples and templated are the base of this project.
2. Stack Overflow for many helpfull code development tips.


