import logging
import os
import re
from datetime import date
from typing import List, Optional, Tuple

import joblib
import pandas as pd
import pytesseract
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageFilter

from .models import Product, PurchaseItem

pytesseract.pytesseract.tesseract_cmd = r"/nix/store/44vcjbcy1p2yhc974bcw250k2r5x5cpa-tesseract-5.3.4/bin/tesseract"

logger = logging.getLogger(__name__)

model_path = "core/ml_models/gradient_boosting_model.pkl"
scaler_path = "core/ml_models/scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

feature_order = [
    'quantity', 'price', 'days_since_last_purchase', 'total_spent',
    'purchase_rate', 'value_score'
]


def predict_days_until_runout_from_features(features_dict):
    try:
        features = [features_dict.get(f) for f in feature_order]
        if None in features:
            return None
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)[0]
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

    total_quantity = sum(i.quantity for i in items)
    total_spent = sum(float(i.price) * float(i.quantity) for i in items)
    total_purchases = items.count()

    purchase_rate = total_purchases / days_since_last if days_since_last > 0 else 0

    avg_price = total_spent / total_quantity if total_quantity > 0 else 0
    value_score = avg_price * total_quantity

    return {
        "quantity": latest_purchase.quantity,
        "price": float(latest_purchase.price),
        "days_since_last_purchase": days_since_last,
        "total_spent": total_spent,
        "purchase_rate": purchase_rate,
        "value_score": value_score
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
                # Extract fractional part only
                fractional_part = predicted_days - int(predicted_days)
                # To avoid zero fractional part (e.g., 7.0), if fractional part is 0, fallback to 0.01 (about 15 minutes)
                if fractional_part == 0:
                    fractional_part = 0.01

                # Scale fractional part to max 7 days
                scaled_days = fractional_part * 7

                product_predictions.append({
                    "product_name":
                    product.name,
                    "predicted_days_until_runout":
                    round(scaled_days, 2),
                    "features":
                    features
                })

    return product_predictions


def process_receipt_image(
        image_path: str
) -> Tuple[pd.DataFrame, str, Optional[float], List[str]]:

    failure_reasons = []

    try:
        if not os.path.exists(image_path):
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
