

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

import traceback

DATA_PATH = 'static/data/'
TXT_PATH = 'static/reftxtpapers/overall/'

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




Survey_n_clusters = {
    '2742488' : 3,
    '2830555' : 3,
    '2907070' : 3,
    '3073559' : 3,
    '3274658' : 2
}

Global_survey_id = ""
Global_ref_list = []
Global_category_description = []
Global_category_label = []
Global_df_selected = ""


from demo.taskDes import absGen, introGen, methodologyGen, conclusionGen
from demo.category_and_tsne import clustering, get_cluster_description


class reference_collection(object):
    def __init__(
            self,
            input_df
    ):
        self.input_df = input_df

    def full_match_with_entries_in_pd(self, query_paper_titles):
        entries_in_pd = self.input_df.copy()
        entries_in_pd['ref_title'] = entries_in_pd['ref_title'].apply(str.lower)
        query_paper_titles = [i.lower() for i in query_paper_titles]

        # matched_entries = entries_in_pd[entries_in_pd['ref_title'].isin(query_paper_titles)]
        matched_entries = self.input_df[entries_in_pd['ref_title'].isin(query_paper_titles)]
        #print(matched_entries.shape)
        return matched_entries,matched_entries.shape[0]

    # select the sentences that can match with the topic words
    def match_ref_paper(self, query_paper_titles,match_mode='full', match_ratio=70):

        # query_paper_title = query_paper_title.lower()
        # two modes for str matching
        if match_mode == 'full':
            matched_entries, matched_num = self.full_match_with_entries_in_pd(query_paper_titles)
        return matched_entries, matched_num

def index(request):
    return render(request, 'demo/index.html')


@csrf_exempt
def upload_refs(request):
    import time
    time.sleep(1)
    topic = request.POST.get('topic')
    files = request.POST.getlist('files[]')
    global Global_survey_id
    Global_survey_id = '001'
    Survey_dict[Global_survey_id] = topic
    Survey_Topic_dict[Global_survey_id] = [topic.lower()]

    Survey_n_clusters[Global_survey_id] = 2

    input_folder_dir = DATA_PATH + 'merge.tsv'
    input_tsv = pd.read_csv(input_folder_dir, sep='\t', header=0)
    query_paper_titles = [i.split('.')[0] for i in files]



    ref_set = reference_collection(input_tsv)
    matched_entries_pd, matched_entries_num = ref_set.match_ref_paper(query_paper_titles, match_mode='full')

    # print(matched_entries_pd)
    matched_entries_pd.to_csv(DATA_PATH + '001.tsv', sep='\t')

    references, ref_links, ref_ids = get_refs(Global_survey_id)

    ref_links = [TXT_PATH+i for i in files]

    for i in ref_links:
        print(i)

    ref_list = {
        'references': references,
        'ref_links': ref_links,
        'ref_ids': ref_ids
    }

    ref_list = json.dumps(ref_list)
    return HttpResponse(ref_list)

    # print(files)
    # print(request.POST['files[]'])


    # references, ref_links, ref_ids = get_refs(topic)
    # global Global_survey_id
    # Global_survey_id = topic
    # ref_list = {
    #     'references' : references,
    #     'ref_links'  : ref_links,
    #     'ref_ids'    : ref_ids
    # }
    # ref_list = json.dumps(ref_list)
    # return HttpResponse(ref_list)


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
def automatic_taxonomy(request):
    ref_dict = dict(request.POST)
    print(ref_dict)
    ref_list = ref_dict['refs']
    global Global_ref_list
    Global_ref_list = ref_list

    colors, category_label, category_description =  Clustering_refs(n_clusters=Survey_n_clusters[Global_survey_id])
    global Global_category_description
    Global_category_description = category_description
    global Global_category_label
    Global_category_label = category_label

    cate_list = {
        'colors': colors,
        'category_label': category_label,
        'survey_id': Global_survey_id
    }
    cate_list = json.dumps(cate_list)
    return HttpResponse(cate_list)


@csrf_exempt
def get_survey(request):
    survey_dict = get_survey_text()
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


def get_survey_text(refs=Global_ref_list):
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
            "This is the proceeding",
            [{"subtitle": "This is the first subtitle", "content": "test "*500},
             {"subtitle": "This is the second subtitle", "content": "test "*500},
             {"subtitle": "This is the third subtitle", "content": "test "*500}]
        ],
        'Conclusion': "test "*150,
        'References': []
    }

    try:
        ## abs generation
        abs, last_sent = absGen(Global_survey_id, Global_df_selected, Global_category_label)
        survey['Abstract'] = [abs, last_sent]

        ## Intro generation
        intro = introGen(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        survey['Introduction'] = intro

        ## Methodology generation
        proceeding, detailed_des = methodologyGen(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        survey['Methodology'] = [proceeding, detailed_des]

        ## Conclusion generation
        conclusion = conclusionGen(Global_survey_id, Global_category_label)
        survey['Conclusion'] = conclusion

        ## reference
        ## here is the algorithm part
        # df = pd.read_csv(data_path, sep='\t')
        for ref in Global_df_selected['ref_entry']:
            entry = str(ref)
            survey['References'].append(entry)

    except:
        print(traceback.print_exc())
    return survey


def Clustering_refs(n_clusters):
    df = pd.read_csv(DATA_PATH + Global_survey_id + '.tsv', sep='\t', index_col=0)
    df_selected = df.iloc[Global_ref_list]

    print(df_selected.shape)
    ## update cluster labels and keywords
    df_selected, colors = clustering(df_selected, n_clusters, Global_survey_id)
    # print(colors)
    print(df_selected.shape)
    ## get description and topic word for each cluster
    description_list = get_cluster_description(df_selected, Global_survey_id)
    # print(description_list)

    global Global_df_selected
    Global_df_selected = df_selected
    category_description = [0]*len(colors)
    category_label = [0]*len(colors)
    for i in range(len(colors)):
        for j in description_list:
            if j['category'] == i:
                category_description[i] = j['category_desp']
                category_label[i] = j['topic_word'].replace('-', ' ').title()
    return colors, category_label, category_description
