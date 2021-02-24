from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('',views.index,name="index"),
    path('fileupload/',views.fileupload,name="fileupload"),
    path('filedelete/<str:f_id>',views.filedelete,name="filedelete"),
    path('single/<str:f_id>',views.single,name="single"),
    path('classification/<str:f_id>',views.classification,name="classification"),
    path('regression/<str:f_id>',views.regression,name="regression")
]