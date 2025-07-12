from django.apps import AppConfig
from django.contrib.auth.models import User
from django.db.utils import OperationalError, ProgrammingError
import csv
import os

class CoreConfig(AppConfig):
    name = 'core'

    def ready(self):
        # This code runs when Django starts
        # Wrap in try/except to avoid issues when migrations haven't run yet
        try:
            # Create superuser if not exists
            if not User.objects.filter(username='admin').exists():
                User.objects.create_superuser(
                    username=os.getenv('DJANGO_ADMIN_NAME','admin',
                    email=os.getenv('DJANGO_ADMIN_EMAIL','admin@email.com'),
                    password=os.getenv('DJANGO_ADMIN_PASSWORD', 'adminpass123')
                )
                print("Superuser created")

            # Create test user if not exists
            if not User.objects.filter(username='testuser').exists():
                User.objects.create_user(
                    username='testuser',
                    email='testuser@example.com',
                    password='testpass123'
                )
                print("Test user created")

            # Load CSV data for test user (example)
            from your_app.models import Purchase  # adjust import to your model
            if Purchase.objects.filter(user__username='testuser').count() == 0:
                csv_path = os.path.join(os.path.dirname(__file__), 'data.csv')
                if os.path.exists(csv_path):
                    test_user = User.objects.get(username='testuser')
                    with open(csv_path, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            # Example: adapt based on your Purchase model fields
                            Purchase.objects.create(
                                user=test_user,
                                item=row['item'],
                                price=float(row['price']),
                                date=row['date'],
                            )
                    print("CSV data imported for testuser")
        except (OperationalError, ProgrammingError):
            # Database not ready, migrations not applied yet
            pass
