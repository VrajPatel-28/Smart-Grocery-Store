from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import CustomUser, Product, Purchase, PurchaseItem, Store

admin.site.register(Store)
admin.site.register(Product)
admin.site.register(Purchase)
admin.site.register(PurchaseItem)
admin.site.register(CustomUser, UserAdmin)
