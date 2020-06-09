

from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import os
import json
import requests
import pandas as pd

DATA_PATH = 'static/data/'

def index(request):
    return render(request, 'demo/index.html')

@csrf_exempt
def get_topic(request):
    topic = request.POST.get('topics', False)
    references, ref_links, ref_ids = get_refs(topic)
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
        ref_path   = os.path.join(DATA_PATH, topic + '.csv')
        df         = pd.read_csv(ref_path)
        references = list(df['ref_title'])
        ref_links  = list(df['ref_link'])
        ref_ids    = list(df['ref_id'])
    except:
        references = default_references
        ref_links = default_ref_links
        ref_ids = default_ref_ids
    return references, ref_links, ref_ids


def get_survey_text(refs):
    '''
    Get the survey text from a given ref list
    Return with a dict as below default value:
    '''
    default_survey = {
        'Title': "This is the survey title",
        'Abstract': "test "*150,
        'Introduction': "test "*500,
        'Methodology': [
            {"subtitle": "This is the first subtitle", "content": "test "*500},
            {"subtitle": "This is the second subtitle", "content": "test "*500},
            {"subtitle": "This is the third subtitle", "content": "test "*500}
        ],
        'Conclusion': "test "*150
    }
    survey = {}
    try:
        ## here is the algorithm part
        print(survey['Title'])
    except:
        survey = default_survey
    return survey