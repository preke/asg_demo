from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    url(r'^get_topic$', views.get_topic, name='get_topic'),
    url(r'^get_survey$', views.get_survey, name='get_survey'),
    url(r'^automatic_taxonomy$', views.automatic_taxonomy, name='automatic_taxonomy'),
    url(r'^upload_refs$', views.upload_refs, name='upload_refs'),
    url(r'^select_sections$', views.select_sections, name='select_sections'),
]