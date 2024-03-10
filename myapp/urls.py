from django.urls import path
from . import views

urlpatterns = [
    path('',views.index),
    path('welcome/',views.welcome, name='welcome'),
    path('preprocess/',views.preprocess,name='preprocess'),
    path('html1/', views.html1, name='html1'),
    path('loadhtml/',views.loadhtml,name='loadhtml'),
    path('about/', views.about, name='about'),
    path('feedback/', views.feedback, name='feedback'),
    path('capture/',views.capture,name='capture'),
    path('detectemotion/',views.detectemotion,name='detectemotion'),
    ]