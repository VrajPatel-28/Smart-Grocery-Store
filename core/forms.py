from django import forms

from .models import Receipt


class ReceiptForm(forms.ModelForm):

    class Meta:
        model = Receipt
        fields = ["purchase", "image"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['purchase'].required = False

from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm

User = get_user_model()

class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = User  # your custom user
        fields = ('username', 'email')