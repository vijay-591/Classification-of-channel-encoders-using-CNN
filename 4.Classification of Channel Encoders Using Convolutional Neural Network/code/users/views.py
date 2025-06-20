from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.core.files.storage import FileSystemStorage
from matplotlib import pyplot as plt
import os
from django.conf import settings


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def ViewDataset(request):
    import pandas as pd
    path = os.path.join(settings.MEDIA_ROOT, "Dataset.csv")
    df = pd.read_csv(path,header=0)
    print(df.head())
    df = df.head(100).to_html(index=False)
    return render(request, 'users/user_view_data.html', {'data': df})


def sampleencodestring(request):
    if request.method == 'POST':
        msgText = request.POST.get('msgText')
        from .utility.test_encoders_strings import start_process
        result = start_process(msgText)
        return render(request, 'users/test_result.html', {'result': result})

    else:
        return render(request, 'users/test_encode.html', {})


def cnnModelTest(request):
    from .utility.ChannelEncodercnn import start_process
    result = start_process()
    return render(request, 'users/results.html', {})