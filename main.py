from flask import Flask, jsonify, request
import json
import requests
from tools import do512input_verifier,process_df,parse_input,MLpredict, get_total_nevt
from voluptuous import MultipleInvalid, Invalid
import boto3 # AWS
import botocore
from datetime import date

#from Data_types import data_types
predict_api = Flask('__name__')

@predict_api.route('/')
def index():
    return 'Hello world! This route works.'

# tester0
@predict_api.route('/post/<int:post_id>', methods=['POST'])
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id

#tester 1
@predict_api.route("/upload/", methods=['POST'])
def upload():
    if request.json:
        return jsonify(request.json)
    else:
        return "Invalid event specs.", 400
    return 'sucess post.'

@predict_api.route("/predict/", methods=['POST'])
def predict():
    
    if request.json:
        r = request.get_json()
        indata = request.json
        # get the number of events, loop through them and then validate
        nevt = get_total_nevt(indata)


        for i in range(nevt):
        # validate dictionary data content: True if has all components, false if not, throw invalid response ]
        # this part can be dropped if once the eventful backend had implimented data validator
            exc= None
            try:
                do512input_verifier(indata['events'][i])
                #raise AssertionError('MultipleInvalid not raised')
            except MultipleInvalid as e:
                exc = e
        
            if exc:
                return "Invalid event info.", 400
       
        # everything went through, call data formator, pass on all events and then return dataframe
        df = parse_input(indata)

        # call data cleaning/transform
        df = process_df(df)
        # save a copy of the transformed data to AWS S3
        ##... 
        # get current time
        today = date.today()
        cfname = 'pred_data_'+str(today)+'.csv'
        df.to_csv(cfname,index=None)

        BucketN = 'evenful-devops'
        Key = 'ML_pred_data_archive/'
        df.to_csv(cfname,index=False)

        client = boto3.client('s3') 
        # Upload the file
        try:
           response = client.upload_file(cfname, BucketN, Key+cfname)
        except ClientError as e:
           logging.error(e)


        # call the prediction phase
        pred = MLpredict(df)
        

        return jsonify(pred), 200
    else:
        return "Invalid event info-- not json.", 400

