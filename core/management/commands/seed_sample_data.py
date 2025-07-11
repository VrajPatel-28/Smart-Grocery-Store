# core/management/commands/seed_sample_data.py

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import Product, Purchase, PurchaseItem, Store

User = get_user_model()


class Command(BaseCommand):
    help = "Seed sample data for development"

    def handle(self, *args, **options):
        # Create user
        user, _ = User.objects.get_or_create(
            username="sampleuser", defaults={"email": "sample@example.com"})
        user.set_password("password123")
        user.save()

        # Create stores
        store1, _ = Store.objects.get_or_create(
            name="SuperMart", defaults={"address": "123 Market Street"})
        store2, _ = Store.objects.get_or_create(
            name="FreshFoods", defaults={"address": "456 Fresh Ave"})

        # Create products
        product1, _ = Product.objects.get_or_create(name="Milk")
        product2, _ = Product.objects.get_or_create(name="Bread")
        product3, _ = Product.objects.get_or_create(name="Eggs")

        # Create purchases
        purchase1 = Purchase.objects.create(user=user,
                                            store=store1,
                                            date=timezone.now().date(),
                                            total_amount=50.00)
        purchase2 = Purchase.objects.create(user=user,
                                            store=store2,
                                            date=timezone.now().date(),
                                            total_amount=30.00)

        # Purchase items
        PurchaseItem.objects.create(purchase=purchase1,
                                    product=product1,
                                    quantity=2,
                                    price=10.00)
        PurchaseItem.objects.create(purchase=purchase1,
                                    product=product2,
                                    quantity=1,
                                    price=30.00)
        PurchaseItem.objects.create(purchase=purchase2,
                                    product=product3,
                                    quantity=3,
                                    price=10.00)

        self.stdout.write(
            self.style.SUCCESS("Sample data created successfully."))
