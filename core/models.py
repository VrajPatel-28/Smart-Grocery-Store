from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    """
    Extend Django's built-in user.
    Add fields here in future (e.g., default store preference).
    """
    pass


class Receipt(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE,
                             default=1)
    purchase = models.ForeignKey('Purchase',
                                 on_delete=models.CASCADE,
                                 null=True,
                                 blank=True)
    image = models.ImageField(upload_to='')
    store_name = models.CharField(max_length=255, blank=True)
    total = models.FloatField(null=True, blank=True)
    extracted_data = models.JSONField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True)

    def __str__(self) -> str:
        store = str(self.store_name) or "Receipt"
        if self.uploaded_at:
            return f"{store} - {self.uploaded_at.strftime('%Y-%m-%d')}"  # type: ignore
        return store


class Store(models.Model):
    name = models.CharField(max_length=200)
    address = models.TextField(blank=True)

    def __str__(self):
        return self.name


class Product(models.Model):
    name = models.CharField(max_length=200)
    barcode = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return self.name


class Purchase(models.Model):
    store = models.ForeignKey(Store, on_delete=models.CASCADE, null=True)
    date = models.DateField()
    total_amount = models.FloatField()
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE)


class PurchaseItem(models.Model):
    purchase = models.ForeignKey(Purchase,
                                 on_delete=models.CASCADE,
                                 related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.FloatField()
    price = models.DecimalField(max_digits=10, decimal_places=2)

    expiry_date = models.DateField(null=True, blank=True)

    def __str__(self):
        return f"{self.product.name} × {self.quantity} @ {self.price}"
