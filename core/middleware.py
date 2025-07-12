from django.contrib.auth import login
from django.contrib.auth.models import User
from django.utils.deprecation import MiddlewareMixin

class AutoLoginTestUserMiddleware(MiddlewareMixin):
    def process_request(self, request):
        # Only auto-login if the user is anonymous AND no "manual_login" flag in session
        if not request.user.is_authenticated and not request.session.get('manual_login'):
            try:
                test_user = User.objects.get(username='testuser')
                login(request, test_user)
            except User.DoesNotExist:
                pass  # testuser not created yet
