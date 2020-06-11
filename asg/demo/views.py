

from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import os
import json
import requests
import time
import pandas as pd
import numpy as np
from demo.taskDes import sumAllSent
import traceback

DATA_PATH = 'static/data/'

Survey_dict = {
    '2742488' : 'Energy Efficiency in Cloud Computing',
    '2830555' : 'Cache Management for Real-Time Systems',
    '2907070' : 'Predictive Modeling on Imbalanced Data',
    '3073559' : 'Malware Detection with Data Mining',
    '3274658' : 'Analysis of Handwritten Signature'
}

Survey_Topic_dict = {
    '2742488' : ['energy'],
    '2830555' : ['cache'],
    '2907070' : ['imbalanced'],
    '3073559' : ['malware', 'detection'],
    '3274658' : ['handwritten', 'signature']
}

Global_survey_id = ""

def index(request):
    return render(request, 'demo/index.html')

@csrf_exempt
def get_topic(request):
    topic = request.POST.get('topics', False)
    references, ref_links, ref_ids = get_refs(topic)
    global Global_survey_id
    Global_survey_id = topic
    ref_list = {
        'references' : references,
        'ref_links'  : ref_links,
        'ref_ids'    : ref_ids
    }
    ref_list = json.dumps(ref_list)
    return HttpResponse(ref_list)

@csrf_exempt
def get_survey(request):
    refs = request.POST.get('refs', False)
    ref_dict = dict(request.POST)
    ref_list = ref_dict['refs']
    print(len(ref_list))
    survey_dict = get_survey_text(ref_list)
    survey_dict = json.dumps(survey_dict)
    return HttpResponse(survey_dict)


def get_refs(topic):
    '''
    Get the references from given topic
    Return with a list
    '''
    default_references = ['ref1','ref2','ref3','ref4','ref5','ref6','ref7','ref8','ref9','ref10']
    default_ref_links = ['', '', '', '', '', '', '', '', '', '']
    default_ref_ids = ['', '', '', '', '', '', '', '', '', '']
    references = []
    ref_links = []
    ref_ids = []

    try:
        ## here is the algorithm part
        ref_path   = os.path.join(DATA_PATH, topic + '.tsv')
        df         = pd.read_csv(ref_path, sep='\t')
        for i,r in df.iterrows():
            # print(r['intro'], r['ref_title'], i)
            if not pd.isnull(r['intro']):
                references.append(r['ref_title'])
                ref_links.append(r['ref_link'])
                ref_ids.append(i)


    except:
        print(traceback.print_exc())
        references = default_references
        ref_links = default_ref_links
        ref_ids = default_ref_ids
    print(len(ref_ids))
    return references, ref_links, ref_ids


def get_survey_text(refs):
    '''
    Get the survey text from a given ref list
    Return with a dict as below default value:
    '''
    # print(refs)
    survey = {
        'Title': "A Survey of " + Survey_dict[Global_survey_id],
        'Abstract': "test "*150,
        'Introduction': "test "*500,
        'Methodology': [
            {"subtitle": "This is the first subtitle", "content": "test "*500},
            {"subtitle": "This is the second subtitle", "content": "test "*500},
            {"subtitle": "This is the third subtitle", "content": "test "*500}
        ],
        'Conclusion': "test "*150,
        'References': []
    }
    data_path = os.path.join(DATA_PATH, Global_survey_id + '.tsv')
    try:
        ## abs generation
        abs = sumAllSent(Survey_Topic_dict[Global_survey_id], data_path)
        survey['Abstract'] = abs
        ## reference
        ## here is the algorithm part
        df = pd.read_csv(data_path, sep='\t')
        for ref in refs:
            entry = str(df.loc[int(ref)]['ref_entry'])
            survey['References'].append(entry)
    except:
        print(traceback.print_exc())
    return survey