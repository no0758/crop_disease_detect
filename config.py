import os
import json


width = 224
height = 224
path = os.path.dirname(os.getcwd())+'/ai_challenger_data/data/'
train_file = path+'train_set/AgriculturalDisease_trainingset/'
test_file = path+'valid_set/AgriculturalDisease_validationset/'

train_path = train_file + 'images/'
train_data_path = train_file + 'AgriculturalDisease_train_annotations.json'
test_path = test_file + 'images/'
test_data_path = test_file + 'AgriculturalDisease_validation_annotations.json'

train_data = json.load(open(train_data_path,'rb'))
test_data = json.load(open(test_data_path,'rb'))

batch_size = 128