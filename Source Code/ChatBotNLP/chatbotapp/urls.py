from django.conf.urls import url

from chatbotapp import ChatBotVietNamese
from . import views
urlpatterns = [
    url('^/?$', views.chatbotvietnamese, name="chatbot"),
    url('^login/user/?$', views.login_func, name="login_func"),
    url('^logout/?$', views.logout_func, name="logout_func"),
    url('^signup/?$', views.signup_func, name="signup_func"),
    url('^updateform/?$', views.update_form, name="updateform"),
    url('^api/vietnamese/?(?P<pk>[\w|\W]+)?$', ChatBotVietNamese.APIChatVietNamese.as_view(), name="chatvietnamese"),
    url('^chatbot/?$', views.chatbotvietnamese, name="chatbot"),
    url('^error/?$', views.error, name="error"),
    url('^updateInfo/?$', views.update_info, name="update"),
]

handler404 = "chatbotapp.views.handle_not_found"