import hashlib
import json
import logging
import os
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
import pytesseract
from django.core.cache import cache
from django.core.files.storage import default_storage
from django.db.models import ExpressionWrapper, F, FloatField, Sum
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageFilter

from .models import Product, PurchaseItem

# Configure logging
logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = r"/nix/store/44vcjbcy1p2yhc974bcw250k2r5x5cpa-tesseract-5.3.4/bin/tesseract"

model_path = "core/ml_models/gradient_boosting_model.pkl"
model = joblib.load(model_path)

feature_order = [
    'quantity', 'price', 'days_since_last_purchase', 'total_spent',
    'purchase_rate', 'value_score'
]


def predict_days_until_runout_from_features(features_dict):
    try:
        features = [features_dict.get(f) for f in feature_order]

        if None in features:
            return None

        # Directly predict without scaling
        expiry_days = features_dict.get("expiry_days", 0)
        prediction = model.predict([
            features
        ])[0] + expiry_days  # Pass as a 2D array for compatibility

        return round(float(prediction), 2)
    except Exception:
        return None


def get_features_for_product(user, product):
    items = PurchaseItem.objects.filter(
        purchase__user=user, product=product).order_by('-purchase__date')

    if not items.exists():
        return None

    latest_purchase = items.first()
    today = date.today()
    last_purchase_date = latest_purchase.purchase.date
    days_since_last = (today - last_purchase_date).days

    # Aggregate data
    total_quantity = items.aggregate(
        total_qty=Sum('quantity'))['total_qty'] or 0

    total_spent = items.annotate(total_line=ExpressionWrapper(
        F('price') * F('quantity'), output_field=FloatField())).aggregate(
            total_spent=Sum('total_line'))['total_spent'] or 0

    total_purchases = items.count()
    purchase_rate = total_purchases / days_since_last if days_since_last > 0 else 0
    avg_price = total_spent / total_quantity if total_quantity > 0 else 0
    value_score = avg_price * total_quantity

    # Handle expiry
    if latest_purchase.expiry_date is None:
        expiry_days = 1
    else:
        expiry_days = (latest_purchase.expiry_date - today).days

    expiry_days = max(expiry_days, 0)

    # Optional: update expiry
    if expiry_days > 0:
        latest_purchase.expiry_date = today + timedelta(days=expiry_days)
        latest_purchase.save()

    return {
        "quantity": total_quantity,
        "price": avg_price,
        "days_since_last_purchase": days_since_last,
        "total_spent": total_spent,
        "purchase_rate": purchase_rate,
        "value_score": value_score,
        "expiry_days": expiry_days,
        "last_purchase_date": last_purchase_date.strftime('%Y-%m-%d')
    }


def get_product_runout_predictions(user):
    product_predictions = []
    purchased_products = Product.objects.filter(
        purchaseitem__purchase__user=user).distinct()

    for product in purchased_products:
        features = get_features_for_product(user, product)

        if features:
            predicted_days = predict_days_until_runout_from_features(features)

            if predicted_days is not None:
                predicted_days += features["expiry_days"]

                fractional_part = predicted_days - int(predicted_days)

                if fractional_part == 0:
                    fractional_part = 0.01  # If zero, to avoid it being treated as an integer

                scaled_days = fractional_part * 7

                product_predictions.append({
                    "product_name":
                    product.name,
                    "predicted_days_until_runout":
                    round(scaled_days, 2),
                    "features":
                    features,
                    "last_purchase_date":
                    features["last_purchase_date"]
                })

    return product_predictions


def calculate_product_runout(product):
    # Implement your logic for calculating how many days until this product runs out
    return 1  # Placeholder for actual logic


def process_receipt_image(
        image_path: str
) -> Tuple[pd.DataFrame, str, Optional[float], List[str]]:

    failure_reasons = []

    try:
        if not default_storage.exists(os.path.basename(image_path)):
            msg = f"Image path does not exist: {image_path}"
            logger.error(msg)
            failure_reasons.append(msg)
            return pd.DataFrame(), "UNKNOWN STORE", None, failure_reasons

        logger.info(f"Processing receipt image: {image_path}")
        img = Image.open(image_path).convert('L')
        logger.debug("Image converted to grayscale")

        img = img.filter(ImageFilter.SHARPEN)
        logger.debug("Image sharpened")

        img = ImageEnhance.Contrast(img).enhance(2.0)
        logger.debug("Image contrast enhanced")

        raw_text = pytesseract.image_to_string(img, lang='eng')
        logger.debug("OCR performed on image")

        items_df, store_name, total_amount = parse_receipt(raw_text)
        logger.info(
            f"Parsed receipt: store_name={store_name}, total_amount={total_amount}"
        )

        if store_name == "UNKNOWN STORE":
            failure_reasons.append("Store name not detected")
        if total_amount is None:
            failure_reasons.append("Total amount not detected")
        if items_df.empty:
            failure_reasons.append("No line items detected")

        return items_df, store_name, total_amount, failure_reasons

    except Exception as e:
        msg = f"Exception during OCR: {e}"
        logger.exception(msg)
        failure_reasons.append(msg)
    return pd.DataFrame(), "UNKNOWN STORE", None, failure_reasons


def detect_store_name(lines: List[str]) -> str:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.search(r'(receipt|bill|ship to|date|#\s*\w)', line, re.I):
            logger.debug(f"Skipping line: {line}")
            continue
        logger.debug(f"Detected store name: {line}")
        return line
    logger.warning("Unable to detect store name")
    return "UNKNOWN STORE"


def detect_total(text: str) -> Optional[float]:
    match = re.search(
        r'(?:Total|subtotal|Amount\s+Due|Receipt\s+Total)\s*[:\-]?\s*[₹$€]?\s*([\d,]+(?:\.\d{1,2})?)',
        text, re.IGNORECASE)
    if match:
        total = float(match.group(1).replace(',', ''))
        logger.debug(f"Detected total amount: {total}")
        return total
    logger.warning("Unable to detect total amount")
    return None


def parse_items(text: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_idx = next(
        (i for i, ln in enumerate(lines)
         if re.search(r'\b(description|item|name|product)\b', ln, re.I)), None)

    pattern = re.compile(
        r'^(?P<item>.+?)\s+(?P<qty>\d+(?:[.,]\d+)?)\s+[₹$€]?\s*(?P<price>\d+(?:[.,]\d{1,2})?)$'
    )
    stop_re = re.compile(
        r'\b(total|sub\s*total|amount\s+due|receipt\s+total)\b', re.I)

    search_range = lines[header_idx + 1:] if header_idx is not None else lines

    records = []
    for ln in search_range:
        if stop_re.search(ln):
            break
        match = pattern.match(ln)
        if not match:
            continue
        item = match.group('item').strip()
        qty = float(match.group('qty').replace(',', '.'))
        price = float(match.group('price').replace(',', '.'))
        records.append((item, qty, price))

    return pd.DataFrame(records, columns=["item_name", "quantity", "price"])


def parse_receipt(text: str) -> Tuple[pd.DataFrame, str, Optional[float]]:
    lines = text.splitlines()
    store_name = detect_store_name(lines)
    total_amount = detect_total(text)
    items_df = parse_items(text)
    return items_df, store_name, total_amount


class NVIDIAAIWrapper:

    def __init__(self, model="nvidia/llama-3.1-nemotron-ultra-253b-v1"):
        api_key = os.environ.get("OPEN_API_KEY")
        if not api_key:
            logger.warning("⚠️ NVIDIA_API_KEY not found. Using fallback mode.")
            self.client = None
        else:
            try:
                self.client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=api_key)
                self.model = model
            except Exception as e:
                logger.error(f"⚠️ LLM init failed: {e}")
                self.client = None

    def get_response(self, prompt: str) -> str:
        if not self.client:
            return "⚠️ AI is currently unavailable."
        try:
            resp = self.client.chat.completions.create(model=self.model,
                                                       messages=[{
                                                           "role":
                                                           "user",
                                                           "content":
                                                           prompt
                                                       }])
            return resp.choices[0].message.content.strip(
            ) if resp.choices else "⚠️ Empty AI response."
        except Exception as e:
            return f"⚠️ Error getting AI response: {e}"


# Singleton instance
nvidia_ai = NVIDIAAIWrapper()


def handle_analytics_query(data: List[Dict], query: str) -> Dict[str, Any]:
    print("🔍 Handling analytics query...")

    if not query or not data:
        logger.warning("Missing query or data.")
        return {
            "llm_response": {
                "error": "No data or query provided"
            },
            "chart_data": None,
            "recipe_suggestions": {}
        }

    # 🧹 Step 1: Clean + reduce data
    cleaned_data = [
        {
            "product": item.get("product__name"),
            "quantity": float(item.get("quantity", 0)),
            "total_spent": float(item.get("total_spent", 0))
        } for item in data[:10]  # limit to 10 items max
    ]

    # 🧠 Step 2: Create cache key
    cache_key = hashlib.sha256(
        (json.dumps(cleaned_data, separators=(",", ":")) +
         query).encode("utf-8")).hexdigest()

    cached_result = cache.get(cache_key)
    if cached_result:
        print("⚡ Returning cached analytics result.")
        return cached_result

    result = {"llm_response": "", "chart_data": None, "recipe_suggestions": {}}
    ai = NVIDIAAIWrapper()

    # === CHART GENERATION ===
    try:
        prompt = generate_chart_prompt(cleaned_data, query)
        print("📤 Sending chart prompt...")
        raw = ai.get_response(prompt)
        print("✅ Chart response received")

        cleaned = re.sub(r"<think>.*?</think>", "", raw,
                         flags=re.DOTALL).strip()
        result["llm_response"] = cleaned

        chart_json = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if chart_json:
            result["chart_data"] = json.loads(chart_json.group())
            print("📊 Chart JSON parsed.")
        else:
            print("⚠️ Chart response not JSON.")
            result["chart_data"] = {"error": "Invalid chart format"}

    except Exception as e:
        logger.exception("Chart generation error.")
        result["chart_data"] = {"error": str(e)}

    # === RECIPE SUGGESTIONS ===
    try:
        inventory = [item["product"] for item in cleaned_data]
        recipe_prompt = generate_recipe_prompt(inventory)
        print("📤 Sending recipe prompt...")
        recipe_raw = ai.get_response(recipe_prompt)
        print("✅ Recipe response received")
        cleaned = re.sub(r"<think>.*?</think>",
                         "",
                         recipe_raw,
                         flags=re.DOTALL).strip()
        try:
            result["recipe_suggestions"] = cleaned
            print("🍲 Parsed recipe suggestions.")
        except json.JSONDecodeError:
            print("⚠️ Recipe response not JSON.")
            result["recipe_suggestions"] = {"error": "Non-JSON response"}

    except Exception as e:
        logger.exception("Recipe generation error.")
        result["recipe_suggestions"] = {"error": str(e)}

    # 💾 Cache the result for faster reuse (e.g., 10 mins)
    cache.set(cache_key, result, timeout=600)
    print("✅ Finished handling analytics query. Result cached.\n")
    return result


def generate_chart_prompt(data: List[Dict], query: str) -> str:
    """
    Compact and efficient chart prompt.
    """
    short_data = json.dumps(data, separators=(",", ":"))
    return f"""
Analyze grocery data and generate a chart JSON if visualizable.
DATA:
{short_data}
QUERY:
"{query}"
INSTRUCTIONS:
- If visualizable, return JSON with: "labels", "values", "title", and "type" ("bar", "line", or "pie" based on best fit).
- Prioritize simple, meaningful visualizations; use aggregates like sums or averages if data is small.
- Example: {{"labels": ["Milk", "Eggs"], "values": [5, 2], "title": "Top Purchases", "type": "bar"}}
- If not visualizable, return: {{"error": "No data for visualization"}}
- Keep response minimal and direct.
""".strip()


def generate_recipe_prompt(inventory: List[str]) -> str:
    """
    Efficient prompt for waste-minimizing recipe suggestions.
    """
    items = ", ".join(inventory)
    return f"""
    Hey! I’ve got these items in my kitchen: {items}.

    Can you suggest 1–2 simple, tasty recipes that would help me use them up and avoid food waste?

    Please include:
    - The recipe name
    - Brief steps or ingredients
    - A helpful tip for minimizing waste

    Keep it casual and easy to follow. just plain text, like you’re texting a friend. 😊
    """.strip()
