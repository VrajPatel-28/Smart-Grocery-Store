import json
import logging
import os
import re
from typing import Any, Dict

import joblib
import openai
from django.contrib import messages
from django.contrib.auth import (
    authenticate,
    login,
    update_session_auth_hash,
)
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import (
    PasswordChangeForm, AuthenticationForm)
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Sum
from django.http import HttpResponseRedirect
from django.shortcuts import redirect, render
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.views import View
from django.views.decorators.http import require_http_methods
from django.views.generic import DetailView, ListView, TemplateView
from django.views.generic.edit import CreateView
from django.utils.decorators import method_decorator
from .forms import CustomUserCreationForm, ReceiptForm
from .models import Product, Purchase, PurchaseItem, Receipt, Store
from .utils import (
    get_features_for_product,
    get_product_runout_predictions,
    nvidia_ai,
    predict_days_until_runout_from_features,
    process_receipt_image,
)

openai.api_key = os.environ.get("OPEN_API_KEY")

logger = logging.getLogger(__name__)

model_path = "core/ml_models/gradient_boosting_model.pkl"
scaler_path = "core/ml_models/scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

feature_order = [
    'quantity', 'price', 'days_since_last_purchase', 'total_spent',
    'purchase_rate', 'value_score'
]


class PurchaseListView(LoginRequiredMixin, ListView):
    model = Purchase
    template_name = "base.html"
    context_object_name = "purchases"
    ordering = ["-date"]

    def get_queryset(self):
        return Purchase.objects.filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["view_type"] = "list"
        ctx["total_purchases"] = ctx["purchases"].count()
        ctx["total_spent"] = ctx["purchases"].aggregate(
            total=Sum("total_amount"))["total"] or 0
        return ctx


class PurchaseDetailView(LoginRequiredMixin, DetailView):
    model = Purchase
    template_name = "base.html"

    def get_queryset(self):
        return Purchase.objects.filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["view_type"] = "detail"
        return ctx


class PurchaseCreateView(LoginRequiredMixin, CreateView):
    model = Purchase
    fields = ["store", "date", "total_amount"]
    template_name = "base.html"
    success_url = reverse_lazy("purchase_list")

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["view_type"] = "add"
        return ctx


class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = "base.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        user_purchases = Purchase.objects.filter(user=self.request.user)
        ctx.update({
            "view_type":
            "dashboard",
            "recent_purchases":
            user_purchases.order_by("-date")[:5],
            "total_spent":
            user_purchases.aggregate(total=Sum("total_amount"))["total"] or 0,
            "total_purchases":
            user_purchases.count(),
        })
        return ctx

class ReceiptUploadView(LoginRequiredMixin, CreateView):
    model = Receipt
    form_class = ReceiptForm
    template_name = "base.html"

    def get_success_url(self):
        if hasattr(self.object, 'purchase') and self.object.purchase:
            return reverse('purchase_detail', args=[self.object.purchase.id])
        return reverse_lazy("dashboard")

    def form_valid(self, form):
        form.instance.user = self.request.user
        receipt = form.save(commit=False)

        # Adjust this to match your environment or media path
        image_path = receipt.image.path

        # OCR and data extraction
        items_df, store_name, total_amount, failure_reasons = process_receipt_image(
            image_path)

        # Save receipt
        receipt.store_name = store_name or "Unknown"
        receipt.total = total_amount
        receipt.extracted_data = items_df.to_dict(
            orient="records") if not items_df.empty else []
        receipt.save()

        # Create Store and Purchase
        store_obj, _ = Store.objects.get_or_create(
            name=store_name or "Unknown")
        purchase = Purchase.objects.create(user=self.request.user,
                                           store=store_obj,
                                           date=timezone.now().date(),
                                           total_amount=total_amount or 0.0)
        receipt.purchase = purchase
        receipt.save()

        # Store predictions to display if needed
        predictions = []

        # Create PurchaseItems and predict runout
        for _, row in items_df.iterrows():
            product_obj, _ = Product.objects.get_or_create(
                name=row["item_name"])
            PurchaseItem.objects.create(purchase=purchase,
                                        product=product_obj,
                                        quantity=row["quantity"],
                                        price=row["price"])

            # Predict days_until_runout
            features = get_features_for_product(self.request.user, product_obj)
            if features:
                prediction = predict_days_until_runout_from_features(features)
                if prediction is not None:
                    predictions.append({
                        "product_name": product_obj.name,
                        "predicted_days_until_runout": prediction,
                        "features": features
                    })

        # Show prediction messages
        if predictions:
            msg_lines = [
                f"{p['product_name']}: {p['predicted_days_until_runout']} days"
                for p in predictions
            ]
            messages.info(self.request,
                          "Predictions:\n" + "\n".join(msg_lines))

        # Show warning if OCR was partial
        if failure_reasons:
            messages.warning(
                self.request,
                "OCR partially succeeded: " + "; ".join(failure_reasons))
        else:
            messages.success(self.request,
                             "Receipt uploaded and processed successfully.")

        self.object = receipt
        return HttpResponseRedirect(self.get_success_url())

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx['view_type'] = "upload"
        return ctx


class AnalyticsView(LoginRequiredMixin, TemplateView):
    template_name = "base.html"

    def get_context_data(self, **kwargs) -> Dict[str, Any]:
        ctx = super().get_context_data(**kwargs)
        user = self.request.user
        user_purchases = Purchase.objects.filter(user=user)

        ctx.update({
            "view_type":
            "analytics",
            "recent_purchases":
            user_purchases.order_by("-date")[:5],
            "total_spent":
            user_purchases.aggregate(total=Sum("total_amount"))["total"] or 0,
            "total_purchases":
            user_purchases.count(),
            "product_predictions":
            get_product_runout_predictions(user)
        })

        return ctx

    def post(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)

        query = request.POST.get("query")
        context["llm_query"] = query

        if not query:
            context["error"] = "No query provided."
            return self.render_to_response(context)

        # Get data to send to LLM
        data = list(PurchaseItem.objects.all().values('product__name',
                                                      'quantity', 'price',
                                                      'purchase__date'))

        prompt = f"""
        You are a helpful assistant analyzing grocery purchase data.

        DATA (JSON-like rows):
        {data}

        USER QUESTION:
        "{query}"

        👇 INSTRUCTIONS:
        If the answer can be visualized, return a JSON with two keys: "labels" and "values".
        Example: {{"labels": ["Milk", "Eggs"], "values": [5, 2]}}

        Otherwise, return a short natural language answer.
        """

        raw_response = nvidia_ai.get_response(prompt)

        # Remove <think> tags and clean the output
        cleaned_response = re.sub(r"<think>.*?</think>",
                                  "",
                                  raw_response,
                                  flags=re.DOTALL).strip()
        context["llm_response"] = cleaned_response

        # Try extracting chart data from response
        try:
            match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
            if match:
                parsed_json = json.loads(match.group())
                context["chart_data"] = parsed_json
        except json.JSONDecodeError as e:
            context[
                "llm_response"] += f"\n\n⚠️ Failed to parse chart data: {e}"

        return self.render_to_response(context)


class ProfileView(LoginRequiredMixin, TemplateView):
    template_name = 'base.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Pass user info explicitly
        user = self.request.user
        context['user_info'] = {
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
        }
        # Add flag to switch view type inside base.html
        context['view_type'] = 'profile'
        return context


@login_required
def update_email_view(request):
    if request.method == "POST":
        new_email = request.POST.get('email')
        if new_email:
            request.user.email = new_email
            request.user.save()
            messages.success(request, "Email updated successfully!")
        else:
            messages.error(request, "Please enter a valid email.")
        return redirect('profile')
    return redirect('profile')


@login_required
def update_password_view(request):
    if request.method == 'POST':
        form = PasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            form.save()
            update_session_auth_hash(request, form.user)  # Keep user logged in
            messages.success(request,
                             'Your password was updated successfully.')
        else:
            messages.error(request, 'Please correct the error below.')
        return redirect('profile')
    else:
        return redirect('profile')


class SignupView(View):

    def get(self, request):
        form = CustomUserCreationForm()
        return render(request, "base.html", {"form": form})

    def post(self, request):
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('profile')
        return render(request, "base.html", {"form": form})

@method_decorator(require_http_methods(["GET", "POST"]), name='dispatch')
class CustomLoginView(View):

    def get(self, request):
        form = AuthenticationForm()
        return render(request, 'base.html', {
            'form': form,
            'view_type': 'login',
        })

    def post(self, request):
        form = AuthenticationForm(request, data=request.POST)
        next_url = request.GET.get('next') or request.POST.get('next') or '/'
        
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            
            # Validate next URL to prevent open redirect attacks
            if url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}, require_https=request.is_secure()):
                return redirect(next_url)
            else:
                return redirect('profile')  # fallback
            
        # Invalid form, show login with errors
        return render(request, 'base.html', {
            'form': form,
            'view_type': 'login',
        })
