from django.apps import AppConfig
from django.contrib.auth.models import User
from django.db.utils import OperationalError, ProgrammingError
import csv
import os

class CoreConfig(AppConfig):
    name = 'core'

    def ready(self):
        try:
            # Create superuser if not exists
            if not User.objects.filter(username='admin').exists():
                User.objects.create_superuser(
                    username=os.getenv('DJANGO_ADMIN_NAME', 'admin'),
                    email=os.getenv('DJANGO_ADMIN_EMAIL', 'admin@example.com'),
                    password=os.getenv('DJANGO_ADMIN_PASSWORD', 'adminpass123')
                )
                print("✅ Superuser 'admin' created")

            # Create test user if not exists
            if not User.objects.filter(username='testuser').exists():
                User.objects.create_user(
                    username='testuser',
                    email='testuser@example.com',
                    password='testpass123'
                )
                print("✅ Test user 'testuser' created")

            # Import CSV data for testuser only if data doesn't already exist
            from core.models import Purchase, Store  # Replace with correct import if needed

            test_user = User.objects.get(username='testuser')
            if Purchase.objects.filter(user=test_user).count() == 0:
                csv_path = os.path.join(os.path.dirname(__file__), 'data.csv')
                if os.path.exists(csv_path):
                    with open(csv_path, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            # Modify this block to match your Purchase model fields
                            store, _ = Store.objects.get_or_create(name=row.get('store', 'date' , 'total_amount', 'item_name', 'quantity' , 'price'))
                            Purchase.objects.create(
                                user=test_user,
                                store=store,
                                total_amount=float(row['total_amount']),
                                date=row['date']
                            )
                    print("📥 Test data imported from data.csv")
        except (OperationalError, ProgrammingError):
            print("⚠️ Skipped init setup: DB not ready yet")
