

from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import os
import json
import requests

def index(request):
    return render(request, 'demo/index.html')

@csrf_exempt
def get_topic(request):
    topic = request.POST.get('topics', False)
    references = get_refs(topic)
    ref_list = {'references': references}
    ref_list = json.dumps(ref_list)
    print(ref_list)
    return HttpResponse(ref_list)

@csrf_exempt
def get_survey(request):
    refs = request.POST.get('global_refs', False)
    ref_dict = dict(request.POST)
    ref_list = ref_dict['references[]']
    survey_dict = get_survey_text(refs)
    survey_dict = json.dumps(survey_dict)
    return HttpResponse(survey_dict)


def get_refs(topic):
    '''
    Get the references from given topic
    Return with a list
    '''
    default_references = ['ref1','ref2','ref3','ref4','ref5','ref6','ref7','ref8','ref9','ref10','ref11']
    references = []
    try:
        ## here is the algorithm part
        print(references[1])
    except:
        references = default_references
    print(references)
    return references


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