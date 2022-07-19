from typing import ChainMap
from django.db import models
from django.utils.timezone import now
from django.contrib.auth.models import User

class Register(models.Model):
    class Meta:
        verbose_name = 'Người dùng'
        verbose_name_plural = 'Danh sách người dùng'
    username = models.CharField(max_length=200,verbose_name='Username',primary_key=True)
    holot = models.CharField(max_length=50,verbose_name='Họ lót',null = False,blank=True)
    ten = models.CharField(max_length=50,verbose_name='Tên',null=False,blank=True)
    email = models.CharField(max_length=300,verbose_name="Email",)
    gioitinh = models.CharField(max_length=10,verbose_name='Giới tính',blank=True,null=True)
    diachi = models.CharField(max_length=200,verbose_name='Địa chỉ',null=True)
    addtime = models.DateTimeField(default=now, editable=True,verbose_name="Ngày tạo")
    owner = models.ForeignKey(User, verbose_name='Người tạo', on_delete=models.SET_NULL,null=True,blank=True)

    def __str__(self):
        return f"{self.username}"

class History(models.Model):
    class Meta:
        verbose_name = 'Tin nhắn'
        verbose_name_plural = 'Danh sách lịch sử tin nhắn'
    message_owner = models.ForeignKey(Register, verbose_name='Người nhắn', on_delete=models.SET_NULL, null=True)
    message = models.FloatField(verbose_name='Tin nhắn')
    addtime = models.DateTimeField(default=now, verbose_name='Ngày thêm',editable=True)
    owner = models.ForeignKey(User, verbose_name='Người thêm' ,on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return f"{self.message_owner},{self.message},{self.addtime},{self.owner}"
