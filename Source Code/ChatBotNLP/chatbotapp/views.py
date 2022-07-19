from django.shortcuts import render
from django.views.generic import ListView
from . import models
from django.contrib.auth.decorators import login_required

from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect
from django.contrib.auth.forms import UserCreationForm
from .form import CreateUserForm


def chatbotvietnamese(request):
    if not request.user.is_authenticated:
        return redirect('/login/user/')
    else:
        return render(request, "chatbot/chatbotvietnamese.html")

@login_required(login_url="/login/user")
def update_form(request):
    return render(request, "chatbot/updateprofile.html")

def login_func(request):
    us = request.POST.get('name','')
    ps = request.POST.get('pass','')
    context = {'name':us, 'pass':ps}
    if us and ps:
        user = authenticate(username = us, password = ps)
        if user:
            login(request, user)
            url = request.GET.get('next','/updateform')
            return redirect(url)
        context['msg']="Sai tên đăng nhập hoặc mật khẩu"
    return render(request, "chatbot/login1.html", context)

def signup_func(request):
    form = CreateUserForm()
    if request.method == "POST":
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            url = request.GET.get('next','/')
            return redirect(url)
    context = {'form':form}
    return render(request, "chatbot/signup.html", context)

def logout_func(request):
    logout(request)
    url = request.GET.get('next','/')
    return redirect(url)

def error(request):
    return render(request, "chatbot/error.html")

def update_info(request):
    if request.method == "POST":
        username = request.POST['username']
        holot = request.POST['holot']
        ten = request.POST['ten']
        email = request.POST['email']
        gioitinh = request.POST['gioitinh']
        diachi = request.POST['diachi']
        
        
        if models.Register.objects.filter(username=username).exists():
            return render(request, "chatbot/error.html")
        info = models.Register.objects.create(username = username,
                                              holot = holot,
                                              ten = ten,
                                              email = email,
                                              gioitinh = gioitinh,
                                              diachi = diachi,
                                            )
        info.save()
        return redirect('/chatbot/')
    else:
        return render(request, "chatbot/error.html")

def handle_not_found(request, exception):
    return render(request, "chatbot/error.html")
