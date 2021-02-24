from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import *
from .utils import *
from .models import *
import pandas as pd
# Create your views here.

def index(request):
    files = File.objects.all()
    df_tables = []

    for f in files:
        data = pd.read_csv(f.file)
        data_html = data.to_html(max_rows=10)
        df_tables.append(data_html)
        

    # if request.GET.get('show_csv') == 'show_csv':
    #     print("thi ")
    #     print(data)

    context = {"df_tables":df_tables,"files":files}
    return render(request,'mainapp/index.html',context)

def fileupload(request):

    if request.method=='POST':
        form = UploadFileForm(request.POST,request.FILES)
        if form.is_valid():
            newfile = File(file=request.FILES['uploaded_file'])
            newfile.save()
        # data = pd.read_csv(request.FILES['uploaded_file'])
        # print(data)
        # data.to_csv("main.csv")
        # print("data to csved")
        messages.success(request,'ファイルをアップロードしました')
        return redirect("index")
    else:
        form = UploadFileForm()


    context = {'form':form}
    return render(request,'mainapp/fileupload.html',context)

def filedelete(request,f_id):
    f = File.objects.get(id=f_id)
    if request.method=='POST':
        f.delete()
        messages.warning(request,'ファイルを削除しました。')
        return redirect('index')
    context = {"f":f}
    return render(request,'mainapp/filedelete.html',context)
        
def single(request,f_id):
    f_id = f_id
    f = File.objects.get(id=f_id)
    data = pd.read_csv(f.file)

    data_html = data.to_html(max_rows=10)
    description = data.describe().round(3).to_html()
    correlation_matrix = data.corr().round(3).to_html


    context = {"f_id":f_id,"f":f,"data_html":data_html,"description":description,"correlation_matrix":correlation_matrix}
    return render(request,'mainapp/single.html',context)

def classification(request,f_id):
    f_id = f_id
    f = File.objects.get(id=f_id)
    data = pd.read_csv(f.file)

    colnames = data.columns
    methods = ["LightGBM","GradientBoosting","RandomForest"]
    metrics = ["Accuracy","Precision","Recall"]
    results,chosen_metrics = False, False
    if request.method == "POST":
        cats = request.POST.getlist('cat')
        ords = request.POST.getlist('ord')
        expvars = request.POST.getlist('expvar')
        target = request.POST.get('target')
        chosen_methods = request.POST.getlist('methods')
        chosen_metrics = request.POST.getlist('metrics')
        form_answers = {
            "categorical_variables":cats,
            "ordinal_variables":ords,
            "explanatory_variables":expvars,
            "target_variable":target,
            "chosen_methods":chosen_methods,
            "chosen_metrics":chosen_metrics
        }
        # print(form_answers)
        # print("target is : ",target)            
        if len(expvars) != len(colnames) and target != None:
            X , y = preprocess(data,cats,ords,expvars,target)
            results = categorize(X,y,chosen_methods,chosen_metrics)
            messages.success(request,'分析を行いました。')
        
        else:
            if len(expvars) == len(colnames):
                messages.warning(request,'全変数が説明変数になっています。')
            if target == None:
                messages.warning(request,'被説明変数が選択されていません。')

        # mse_list = [
        # {"method":"LGBM","score":{"Accuracy":0.5,"Precision":0.3,"Recall":0.2}},
        # {"method":"RandomForrest","score":{"Accuracy":0.5,"Precision":0.3,"Recall":0.2}}
        # ]

    context = {
    "f_id":f_id,
    "colnames":colnames,
    "methods":methods,
    "metrics":metrics,
    "data":data,
    "chosen_metrics":chosen_metrics,
    "results":results
    }
    return render(request,'mainapp/classification.html',context)


def regression(request,f_id):
    f_id = f_id
    f = File.objects.get(id=f_id)
    data = pd.read_csv(f.file)

    colnames = data.columns
    methods = ["MultipleLinearRegression"]#,"SupportVectorRegression" ,"PolynomialRegression"]
    metrics = ["MSE","RMSE","MAE"]
    results,chosen_metrics = False, False
    if request.method == "POST":
        cats = request.POST.getlist('cat')
        ords = request.POST.getlist('ord')
        expvars = request.POST.getlist('expvar')
        target = request.POST.get('target')
        chosen_methods = request.POST.getlist('methods')
        chosen_metrics = request.POST.getlist('metrics')
        form_answers = {
            "categorical_variables":cats,
            "ordinal_variables":ords,
            "explanatory_variables":expvars,
            "target_variable":target,
            "chosen_methods":chosen_methods,
            "chosen_metrics":chosen_metrics
        }
        # print(form_answers)
        # print("target is : ",target)            
        if len(expvars) != len(colnames) and target != None:
            X , y = preprocess(data,cats,ords,expvars,target)
            results = regress(X,y,chosen_methods,chosen_metrics)
            messages.success(request,'分析を行いました。')
        
        else:
            if len(expvars) == len(colnames):
                messages.warning(request,'全変数が説明変数になっています。')
            if target == None:
                messages.warning(request,'被説明変数が選択されていません。')

        # mse_list = [
        # {"method":"LGBM","score":{"Accuracy":0.5,"Precision":0.3,"Recall":0.2}},
        # {"method":"RandomForrest","score":{"Accuracy":0.5,"Precision":0.3,"Recall":0.2}}
        # ]

    context = {
    "f_id":f_id,
    "colnames":colnames,
    "methods":methods,
    "metrics":metrics,
    "data":data,
    "chosen_metrics":chosen_metrics,
    "results":results
    }
    return render(request,'mainapp/regression.html',context)


# def do_preprocess(request,f_id):
#     f = File.objects.get(id=f_id)
#     if request.method == 'POST'
#         messages.success(request,'前処理を行いました')
#         return redirect('index')
#     context = {"f":f}
#     return render(request,'mainapp/filedelete.html',context)
               
