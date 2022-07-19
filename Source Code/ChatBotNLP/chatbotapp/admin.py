from django.contrib import admin
from . import models
# Register your models here.
    
class RegisterAdmin(admin.ModelAdmin):
    list_display = ('username','holot','ten','email','gioitinh','diachi','addtime')
    search_fields = ['username','ten']
    list_filter = ('username','ten')
    fieldsets = [
        ("Thông tin chung", {'fields': ['username','holot','ten','email','gioitinh','diachi']}),
        ("Sở hữu", {'fields': [('owner')]}),
        ("Thời gian", {'fields': [('addtime')]}),
    ]

    def get_owner(self, obj):
        if not obj.owner:
            return "N/A"
        return obj.owner.get_full_name()
    get_owner.short_description = 'Sở hữu'

class HistoryAdmin(admin.ModelAdmin):
    list_display = ('message_owner','message','addtime')
    search_fields = ['message_owner','addtime']
    list_filter = ('message_owner','addtime')


    def get_owner(self, obj):
        if not obj.owner:
            return "N/A"
        return obj.owner.get_full_name()
    get_owner.short_description = 'Sở hữu'
    
    # def get_class(self, obj):
    #     if not obj.Register:
    #         return "N/A"
    #     return obj.Register.username
    # get_class.short_description = 'Tin nhắn của'
    
admin.site.register(models.Register, RegisterAdmin)
admin.site.register(models.History, HistoryAdmin)