import csv
from datetime import datetime

from django.core.management.base import BaseCommand

from core.models import Product, Purchase, PurchaseItem, Store


class Command(BaseCommand):
    help = "Bulk import purchases and items from CSV"

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str)
        parser.add_argument('--user_id', type=int, required=True)

    def handle(self, *args, **options):
        path = options['csv_file']
        user_id = options['user_id']
        purchases = {}

        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                key = (row['store'], row['date'], row['total_amount'])

                if key not in purchases:
                    store_obj, _ = Store.objects.get_or_create(
                        name=row['store'])
                    purchase = Purchase.objects.create(
                        store=store_obj,
                        date=datetime.strptime(row['date'], "%Y-%m-%d").date(),
                        total_amount=float(row['total_amount']),
                        user_id=user_id)
                    purchases[key] = purchase
                else:
                    purchase = purchases[key]

                product_obj, _ = Product.objects.get_or_create(
                    name=row['item_name'])

                PurchaseItem.objects.create(purchase=purchase,
                                            product=product_obj,
                                            quantity=float(row['quantity']),
                                            price=float(row['price']))

        self.stdout.write(self.style.SUCCESS('Import complete.'))
