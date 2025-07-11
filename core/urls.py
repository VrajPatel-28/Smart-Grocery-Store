from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from django.contrib.auth.views import LogoutView
from django.urls import path

from .views import (
    AnalyticsView,
    DashboardView,
    ProfileView,
    PurchaseCreateView,
    PurchaseDetailView,
    PurchaseListView,
    ReceiptUploadView,
    SignupView,
    run_sql_query,
    update_email_view,
    update_password_view,
)

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    # path('analytics/query/', run_sql_query, name='ask_ai'),
    path('login/',
         auth_views.LoginView.as_view(template_name='login.html'),
         name='login'),
    path('signup/', SignupView.as_view(), name='signup'),
    path('update-email/', update_email_view, name='update_email'),
    path('update-password/', update_password_view, name='update_password'),
    path('logout/', LogoutView.as_view(next_page='/'), name='logout'),
    path('purchases/', PurchaseListView.as_view(), name='purchase_list'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('purchases/<int:pk>/',
         PurchaseDetailView.as_view(),
         name='purchase_detail'),
    path('purchases/add/', PurchaseCreateView.as_view(), name='purchase_add'),
    path("upload-receipt/", ReceiptUploadView.as_view(),
         name="upload_receipt"),
    path("analytics/", AnalyticsView.as_view(), name="analytics")
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
