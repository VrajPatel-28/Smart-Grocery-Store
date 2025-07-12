from django.apps import AppConfig
import threading

class CoreConfig(AppConfig):
    name = 'core'

    def ready(self):
        from django.db.utils import OperationalError, ProgrammingError
        from django.contrib.auth.models import User
        import os, csv

        # Delay execution to ensure full app readiness (avoids AppRegistryNotReady)
        def setup_initial_data():
            try:
                # Create superuser
                if not User.objects.filter(username='admin').exists():
                    User.objects.create_superuser(
                        username=os.getenv('DJANGO_ADMIN_NAME', 'admin'),
                        email=os.getenv('DJANGO_ADMIN_EMAIL', 'admin@email.com'),
                        password=os.getenv('DJANGO_ADMIN_PASSWORD', 'adminpass123')
                    )
                    print("✅ Superuser created")

                # Create test user
                if not User.objects.filter(username='testuser').exists():
                    User.objects.create_user(
                        username='testuser',
                        email='testuser@example.com',
                        password='testpass123'
                    )
                    print("✅ Test user created")

                # Load CSV data
                from core.models import Purchase  # ✅ Replace `your_app` with your actual app name
                if Purchase.objects.filter(user__username='testuser').count() == 0:
                    csv_path = os.path.join(os.path.dirname(__file__), 'data.csv')
                    if os.path.exists(csv_path):
                        test_user = User.objects.get(username='testuser')
                        with open(csv_path, newline='') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for row in reader:
                                Purchase.objects.create(
                                    user=test_user,
                                    item=row['item'],
                                    price=float(row['price']),
                                    date=row['date'],
                                )
                        print("✅ CSV data imported for testuser")

            except (OperationalError, ProgrammingError):
                # Database might not be ready yet (during first migrate)
                print("⚠️ Skipped setup_initial_data: DB not ready")

        # Use threading to delay execution until after full startup
        threading.Thread(target=setup_initial_data).start()
