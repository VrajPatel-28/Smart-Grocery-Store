from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from django.contrib.auth.views import LogoutView
from django.urls import path
from django.contrib import admin

from .views import (
    AnalyticsView,
    DashboardView,
    ProfileView,
    PurchaseCreateView,
    PurchaseDetailView,
    PurchaseListView,
    ReceiptUploadView,
    SignupView,
    update_email_view,
    update_password_view,
    CustomLoginView
)

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path('admin/', admin.site.urls),
    path('login/',
          CustomLoginView.as_view(template_name='base.html'),
         name='login'),
    path('signup/', SignupView.as_view(), name='signup'),
    path('update-email/', update_email_view, name='update_email'),
    path('update-password/', update_password_view, name='update_password'),
    path('logout/', CustomLogoutView.as_view(), name='logout'),
    path('purchases/', PurchaseListView.as_view(), name='purchase_list'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('purchases/<int:pk>/',
         PurchaseDetailView.as_view(),
         name='purchase_detail'),
    path('purchases/add/', PurchaseCreateView.as_view(), name='purchase_add'),
    path("upload-receipt/", ReceiptUploadView.as_view(),
         name="upload_receipt"),
    path("analytics/", AnalyticsView.as_view(), name="analytics"),
    
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
