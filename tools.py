import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from voluptuous import Required, All, Length, Range, Schema, Coerce
import datetime
import re
import json
from tqdm import tqdm # may be cleaned up
import boto3 # AWS
import botocore
from datetime import date, timezone
import pickle

def do512input_verifier(indata): # indata is a dictionary(each event)
	# check if the input dictionary has all the components which is reqired for ML
                                	# input
	
	do512schema = Schema({ Required('tz_adjusted_begin_date'): Timestamp, 'title': str, Required('category'):All(str,cat_in_list), Required('tz_adjusted_end_date'): Timestamp, Required('past'): bool,Required('is_eventbrite'): bool,Required('is_free'): bool,Required('doors'): bool, Required('sold_out'): bool, Required('ticket_info'): str,Required('artists'): list, Required('venue'):Schema({'id':int, 'popularity': Coerce(float), 'zip': All(int,Range(70000,79000)) }),Required('uid'):int,Required('source'): 'Do512'})

	return do512schema(indata)



# voluptuous data validation Schema, specific for do512
def Timestamp(value):
	return datetime.datetime.fromisoformat(value)


def cat_in_list(value):
    valid_cat = ["Music","Film + TV","Community","Food + Drink","Comedy","Art & Culture","Sports + Activity","DJ's + Parties","Literary","Workshops + Classes","Karaoke + Trivia", "Variety / Other","Theater","Happy Hour + Drink Specials", "LGBTQ+","Activism","Exhibit","Opening","Fashion","Free Week"]
    return (value in valid_cat)

def get_total_nevt(indict):
	# get total number of item from request
	count = len(indict['events'])
	
	return count

def parse_input(indict):
	# to parse the incoming json into python dictionary

	input = indict['events']
	# create empty dataframe
	df = pd.DataFrame(columns=list(input[0].keys()))
	# dump all dictionary data to dataframe
	for line in input:
		df = df.append(line,ignore_index=True)
	return df


def process_df(df):

	nevt = len(df)

	# parse ticket info string
	tkt_str = df.ticket_info.values
	isfree = df.is_free.values
	ages = [0 for x in range(nevt)]

	tktp_min, tktp_max, isfree, ages, allages = parse_ticketinfo(tkt_str,isfree,ages, nevt)
	# insert back cleansed info
	df['is_free'] = isfree
	df['min_age'] = ages
	df['ticket_allages']= allages
	df['ticket_price_low'] = tktp_min
	df['ticket_price_max'] = tktp_max

	# work on venues
	df = parse_venue(df)

	# work on the times features
	df = parse_time(df)

	# work on the artists features
	df = parse_artists(df)

	df = cleanup_df(df)


	return df


def parse_ticketinfo(tkt_str,isfree,ages,nevt):
    tktprc_low = [0 for x in range(nevt)]
    tktprc_max = [0 for x in range(nevt)]
    allages = [False for x in range(nevt)]
    
	#print("parsing ticket info...../n")
    for i,item in enumerate(tkt_str):
        #check 'free'
        if ("free" in item.lower()): isfree[i] = True
        # check child ok
        if ("all ages" in item.lower()): 
            ages[i] = 0
            allages[i] = True
        #replace all whitespace after '$'
        item = re.sub('\$ ','$',item)
        prices = re.findall('(?<=\$)\d*\.?\d*', item)
        # get all lower/upper $number
        if prices: # not [] or None
           if (prices[0].isdigit()) :
              p = [float(x) for x in prices]
              tktprc_low[i] = min(p)
              tktprc_max[i] = max(p)
                    
        # get minimum age
        min_ag = re.findall(r"(\d+)\+",item)
        if min_ag: ages[i] = int(min_ag[0])
        else:ages[i]=None
    return (tktprc_low, tktprc_max, isfree, ages, allages)

def parse_venue(df):
	venuejson = df['venue'].values
	vid =[]
	vpop = []
	vzip = []
	for item in venuejson:
		vid += [int(item['id'])]
		vpop += [float(item['popularity'])]
		vzip += [int(item['zip'])]
	df['venue.id']=vid
	df['venue.popularity']=vpop
	df['venue.zip']=vzip

	totnven=[]
	# fetch the data table 
	client = boto3.client('s3')

	venuedb = 'TotalEventsPerVenueID2018.csv'
#	obj = client.get_object(Body= venuedb, Bucket='eventful-devops', Key='EventsMLModel/'+venuedb)  ##
	venuedb = 'models/'+venuedb
	vdb = pd.read_csv(venuedb, index_col=2)

	tolnev=[]
	venid=df['venue.id'].values
	lvenid=vdb.index.values
	for idn in venid:
		if idn in lvenid:tolnev+=[vdb.loc[idn][1]]
		else:tolnev+=[1]
	df['venue.tol_num_events']=tolnev

	return df

def parse_time(df):

	# enforce the utc tz zone
	df['tz_adjusted_begin_date']=pd.to_datetime(df['tz_adjusted_begin_date'].apply(Timestamp), utc=True)
	df['tz_adjusted_end_date']=pd.to_datetime(df['tz_adjusted_end_date'].apply(Timestamp),utc=True)
	


	df['duration']=(df['tz_adjusted_end_date']-df['tz_adjusted_begin_date']).apply(lambda x:x/ np.timedelta64(1, 'h'))
	df['duration_day']=df['duration'].apply(lambda x:np.ceil(x/24))
	a=df['duration'].values/24
	multid=[True if x >=1 else False for x in a]
	df['multiday']=multid
	# fix duration_day=0
	df=df.replace({'duration':0},2)
	df=df.replace({'duration_day':0},1)

	#set up times
	df['dow']=df['tz_adjusted_begin_date'].dt.dayofweek
	df['doy']=df['tz_adjusted_begin_date'].dt.dayofyear
	df['month']=df['tz_adjusted_begin_date'].dt.month
	df['day']=df['tz_adjusted_begin_date'].dt.day
	df['hour']=df['tz_adjusted_begin_date'].dt.hour

	return df

def parse_artists(df):
	# parse artist information
	# get artist column andef
	e_artist=df.artists.values
	#get artist_popularity
	p_art_avg=[]
	p_art_sum=[]
	p_art_max=[]
	for item in e_artist:
		if len(item)==0:
			p_art_sum+=[0]
			p_art_avg+=[0]
			p_art_max+=[0]
		elif len(item)==1:
			p_art_sum+=[item[0]['popularity']]
			p_art_avg+=[item[0]['popularity']]
			p_art_max+=[item[0]['popularity']]
		else:
			subsum=0
			partmax=0
			for sub in item:
				part=sub['popularity']
				subsum+=part
				if part> partmax: partmax=part
			p_art_sum+=[subsum]
			p_art_avg+=[subsum/len(item)]
			p_art_max+=[partmax]

	df['artist.popularity.sum']=p_art_sum
	df['artist.popularity.avg']=p_art_avg
	df['artist.popularity.max']=p_art_max

	return df

def cleanup_df(df):
	# null data cleanup
	df.min_age=df.min_age.fillna(0)
	df.ticket_price_max=df.ticket_price_max.fillna(0)
	df.ticket_price_low=df.ticket_price_low.fillna(0)

	# encoding booleans
	boolcol = ['past','is_eventbrite','is_free','doors','sold_out','multiday','ticket_allages']
	## convert all boolean columns into 0/1
	for cols in boolcol:
		df[cols]=df[cols].astype(int)

	# mapping the category 
	mapcat={'Music ': 0, 'Comedy': 1, 'Happy Hour + Drink Specials': 2, "DJ's + Parties": 3, 'Film + TV': 4, 'Karaoke + Trivia': 5, 'Workshops + Classes': 6, 'Community': 7, 'Literary': 8, 'Art & Culture': 9, 'Sports + Activities': 10, 'Food + Drink': 11, 'LGBTQ+': 12, 'Theater': 13, 'Variety / Other': 14, 'Activism': 15, 'Exhibit': 16, 'Opening': 17, 'Fashion': 18, 'Free Week': 19}

	for i,line in enumerate(mapcat):
		df.category.replace(line,i,inplace=True)

	# mapping the venue id

	# need to fetch the file from S3
	viddb='venue_id_lib.npy'
	viddb = 'models/'+viddb
	encode_dict=np.load(viddb,allow_pickle=True).item()
	df['venue.id']=df['venue.id'].map(encode_dict)
	
	# in case of Nan

	keylst = list(encode_dict.keys())
	mostcommon=keylst[max(encode_dict.values())]
	fullvn_id=df['venue.id'].values
	nvn_id=[]
	for item in fullvn_id:
		if np.isnan(item):
			nvn_id+=[int(mostcommon+1)]
			mostcommon+=1
		else: nvn_id+=[int(item)]
	df['venue.id']=nvn_id

	# mapping the venue zip

	# need to fetch the file from S3
	vzipid='venue_zip_lib.npy'
	vzipid='models/'+vzipid
	encode_zip_dict=np.load(vzipid,allow_pickle=True).item()
	df['venue.zip']=df['venue.zip'].map(encode_zip_dict)

	# in case of Nan
	keylst = list(encode_zip_dict.keys())
	mostcommon=keylst[max(encode_zip_dict.values())]
	fullvnzp_id=df['venue.id'].values
	nvn_id=[]
	for item in fullvnzp_id:
		if np.isnan(item):
			nvn_id+=[int(mostcommon+1)]
			mostcommon+=1
		else: nvn_id+=[int(item)]
	df['venue.zip']=nvn_id

	return df

def dropfeatures(df):
	drop_lst = ['title','tz_adjusted_begin_date','tz_adjusted_end_date','ticket_info','venue','artists',]
	df.drop(drop_lst, axis=1, inplace=True)
	return df

def get_model(source):
	# fetch the right model package based on the data source and feedback the model
	client = boto3.client('s3') 
	s3_prefx = 'EventsMLmodel/'+source+'/'
	today = datetime.datetime(bytearray=date.today().year, month=date.today().month, today=date.today().day, tzinfo=timezone.utc)
	BucketN = 'eventful-devops'

	model_f=source+'_model'
	scalar_f='MinMaxScalar'

	objList = client.list_objects(Bucket = BucketN)['Contents']
	diff_date = []
	f_prefx =[]
	diff_date_sc =[]
	fs_prefx =[]
	for obj in objList:
		obj_Key = obj['Key']
		if s3_prefx+model_f in obj_Key:
			obj_date = obj['LastModified']
			diff_date +=[today-obj_date]
			f_prefx += [obj_Key]
		elif s3_prefx+scalar_f in obj_Key:
			obj_date = obj['LastModified']
			diff_date_sc +=[today-obj_date]
			fs_prefx += [obj_Key]


		# get the latest model
	min_p, min_t = min(enumerate(diff_date))
	_destPath, model_fname = os.path.split(f_prefx[min_p])
	client.download_file(BucketN, model_fname, _destPath)
	# load model
	file= open(model_fname,'rb')
	model = pickle.load(file)
	
	# get latest scalar files
	min_p, min_t = min(enumerate(diff_date_sc))
	_destPath, scalar_fname = os.path.split(fs_prefx[min_p])
	client.download_file(BucketN, scalar_fname, _destPath)
	# load scalar
	file= open(scalar_fname,'rb')
	scalar = pickle.load(file)

#	if source =='Do512':

#	obj = client.put_object( Body= df.to_csv(cfname,index=False), Bucket='eventful-devops', Key='ML_pred_data_archive/'+cfname)

# temporary fix for local tests
#		mname='finalized_model111819.pkl'
#		mname='models/'+mname
#		file= open(mname,'rb')
#		model = pickle.load(file)

#		sname='MinMaxScalar_111819.pkl'
#		sname='models/'+sname
#		file= open(sname,'rb')
#		scalar = pickle.load(file)

	return model, scalar

def MLpredict(df):

	# takes in source data
	fflist=['category', 'past', 'is_eventbrite', 'is_free', 'doors','sold_out', 'venue.id', 'venue.popularity', 'venue.zip','ticket_allages', 'ticket_price_low', 'ticket_price_max', 'min_age','artist.popularity.sum', 'artist.popularity.avg','artist.popularity.max', 'dow', 'doy', 'month', 'day', 'hour','venue.tol_num_events', 'duration', 'duration_day', 'multiday']

	dff=df[fflist]

	# get source
	source = df['source'][0]
	
	# takes in model
	model , scalar = get_model(source)
	# apply scalar
	X= scalar.transform(dff)
	prd = model.predict(X)
	prdprob = model.predict_proba(X)
	# post processing and form dictionary
	pred={}
	content=[]
	for i in range(len(prd)):
		score= prd[i]+prdprob[i][int(prd[i])]
		evuid = df['uid'][i]
		content+=[{'uid':evuid, 'score':score}]
	pred['events']=content
	return pred