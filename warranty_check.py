import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from decouple import config
from datetime import datetime
from typing import Optional, Dict
import cv2
import numpy as np
from pathlib import Path
import shutil
import re
import logging
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# PyTorch
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# OCR
import easyocr

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"🔥 PyTorch Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    print(f"   Using CPU (GPU not available)")
print(f"{'='*70}\n")

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE WITH ABSOLUTE PATH
# ══════════════════════════════════════════════════════════════════════════════

# Use absolute path to ensure correct database
DB_PATH = Path(__file__).parent / "warranty_validation.db"
DATABASE_URL = config("DATABASE_URL", default=f"sqlite:///{DB_PATH}")

print(f"📂 Database Configuration:")
print(f"   Path: {DB_PATH.resolve()}")
print(f"   URL: {DATABASE_URL}")
print()

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone = Column(String(20))
    address = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    warranties = relationship("Warranty", back_populates="customer")

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    product_name = Column(String(200), nullable=False)
    model_number = Column(String(100))
    manufacturer = Column(String(100))
    category = Column(String(50))
    warranty_period_months = Column(Integer)
    warranties = relationship("Warranty", back_populates="product")

class Warranty(Base):
    __tablename__ = "warranties"
    id = Column(Integer, primary_key=True)
    warranty_number = Column(String(50), unique=True, nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    serial_number = Column(String(100))
    purchase_date = Column(DateTime, nullable=False)
    expiry_date = Column(DateTime, nullable=False)
    warranty_terms = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    customer = relationship("Customer", back_populates="warranties")
    product = relationship("Product", back_populates="warranties")

Base.metadata.create_all(bind=engine)

# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH MODEL
# ══════════════════════════════════════════════════════════════════════════════

class WarrantyClassifierPyTorch(nn.Module):
    def __init__(self):
        super(WarrantyClassifierPyTorch, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)

class WarrantyClassifier:
    def __init__(self, model_path='models/best_model_pytorch.pth'):
        self.model_path = Path(model_path)
        self.device = device
        self.class_names = ['authentic', 'fraudulent']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"📥 Loading PyTorch model from {model_path}...")
        self.model = WarrantyClassifierPyTorch().to(self.device)
        
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ PyTorch model loaded successfully!")
        else:
            print(f"⚠️  Model file not found: {model_path}")
    
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
        
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item()
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'authentic': probabilities[0].item(),
                'fraudulent': probabilities[1].item(),
            },
            'is_authentic': predicted_class == 'authentic',
            'is_fraudulent': predicted_class == 'fraudulent'
        }

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class ImagePreprocessor:
    @staticmethod
    def enhance_contrast(img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img)

    @staticmethod
    def preprocess_enhanced(image_path: str) -> str:
        img = cv2.imread(image_path)
        enhanced = ImagePreprocessor.enhance_contrast(img)
        out_path = str(image_path).replace(".jpg", "_enhanced.jpg").replace(".png", "_enhanced.png")
        cv2.imwrite(out_path, enhanced)
        return out_path

# ══════════════════════════════════════════════════════════════════════════════
# OCR ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class OCREngine:
    def __init__(self):
        print("📥 Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
        print(f"✅ EasyOCR loaded!")

    def extract_text(self, image_path: str) -> str:
        # Try original
        text = self._extract_once(image_path)
        if self._has_warranty_pattern(text):
            return text
        
        # Try enhanced
        try:
            enhanced = ImagePreprocessor.preprocess_enhanced(image_path)
            text2 = self._extract_once(enhanced)
            Path(enhanced).unlink()
            if self._has_warranty_pattern(text2):
                return text2
        except:
            pass
        
        return text or ""
    
    def _extract_once(self, image_path: str) -> str:
        try:
            results = self.reader.readtext(image_path, detail=0)
            return " ".join(results)
        except:
            return ""
    
    def _has_warranty_pattern(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False
        return bool(re.search(r'WTY[A-Z0-9]{3}[A-Z0-9]{8}|[Ww]arranty\s+[Nn]umber', text.upper()))

# ══════════════════════════════════════════════════════════════════════════════
# OCR CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

class OCRCorrector:
    """Fix common OCR misreads: S↔5, Z↔2, O↔0, I↔1"""
    
    @staticmethod
    def correct_warranty_number(text: str) -> str:
        if not text:
            return text
        
        pattern = r'\b(WTY[A-Z0-9]{3}[A-Z0-9]{8,12})\b'
        matches = re.findall(pattern, text.upper())
        
        if not matches:
            return text
        
        corrected_text = text
        
        for match in matches:
            corrected = match
            
            # First 6 chars should be LETTERS
            prefix = corrected[:6]
            prefix = prefix.replace('5', 'S')
            prefix = prefix.replace('0', 'O')
            prefix = prefix.replace('1', 'I')
            prefix = prefix.replace('8', 'B')
            prefix = prefix.replace('2', 'Z')
            
            suffix = corrected[6:]
            corrected = prefix + suffix
            corrected_text = corrected_text.replace(match, corrected)
        
        return corrected_text
    
    @staticmethod
    def correct_serial_number(serial: str) -> str:
        if not serial:
            return serial
        
        corrected = serial.upper()
        
        if corrected.startswith('5N'):
            corrected = 'SN' + corrected[2:]
        
        if corrected.startswith('5NN'):
            corrected = 'SNN' + corrected[3:]
        
        if corrected.startswith('SN'):
            prefix = corrected[:2]
            suffix = corrected[2:].replace('Z', '2')
            corrected = prefix + suffix
        
        return corrected
    
    @staticmethod
    def correct_text(text: str) -> str:
        if not text:
            return text
        return OCRCorrector.correct_warranty_number(text)

# ══════════════════════════════════════════════════════════════════════════════
# FIELD EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

class FieldExtractor:
    @staticmethod
    def extract_all(text: str) -> Dict[str, Optional[str]]:
        fields = {
            "warranty_number": None,
            "serial_number": None,
            "purchase_date": None,
            "expiry_date": None,
            "customer_name": None,
            "product_name": None
        }

        if not text:
            print(f"   ⚠️  No text to extract")
            return fields

        text_clean = " ".join(text.split())
        text_corrected = OCRCorrector.correct_text(text_clean)
        
        print(f"\n   📋 Extracting Fields ({len(text_corrected)} chars)")
        if text_clean != text_corrected:
            print(f"   🔧 Applied OCR corrections")

        # Warranty Number
        wty_patterns = [
            r'[Ww]arranty\s+[Nn]umber[:\s]+([A-Z]{3,6}[A-Z0-9]{8,12})',
            r'\b(WTY[A-Z]{3}[A-Z0-9]{6,10})\b',
            r'\b([A-Z]{5,6}[A-Z0-9]{8,12})\b',
        ]
        for pattern in wty_patterns:
            matches = re.findall(pattern, text_corrected)
            if matches:
                warranty_num = matches[0].upper().strip()
                if len(warranty_num) >= 11 and warranty_num[:3].isalpha():
                    fields["warranty_number"] = warranty_num
                    print(f"      ✅ Warranty: {warranty_num}")
                    break
        
        if not fields["warranty_number"]:
            print(f"      ❌ Warranty: Not found")

        # Serial Number
        serial_patterns = [
            r'[Ss]erial\s+[Nn]umber[:\s]+([A-Z0-9]{8,20})',
            r'\b([S5]N[A-Z0-9]{6,18})\b',
        ]
        for pattern in serial_patterns:
            matches = re.findall(pattern, text_corrected, re.IGNORECASE)
            if matches:
                serial = matches[0].upper().strip()
                serial = OCRCorrector.correct_serial_number(serial)
                if len(serial) >= 8:
                    fields["serial_number"] = serial
                    print(f"      ✅ Serial: {serial}")
                    break

        # Dates
        purchase_match = re.search(r'[Pp]urchase\s+[Dd]ate[:\s]+(\d{1,2}/\d{1,2}/\d{4})', text_clean)
        if purchase_match:
            fields["purchase_date"] = purchase_match.group(1)
            print(f"      ✅ Purchase: {fields['purchase_date']}")

        expiry_match = re.search(r'[Ee]xpiry\s+[Dd]ate[:\s]+(\d{1,2}/\d{1,2}/\d{4})', text_clean)
        if expiry_match:
            fields["expiry_date"] = expiry_match.group(1)
            print(f"      ✅ Expiry: {fields['expiry_date']}")

        # Customer Name
        customer_match = re.search(r'[Cc]ustomer\s+[Nn]ame[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text_clean)
        if customer_match:
            fields["customer_name"] = customer_match.group(1).strip()
            print(f"      ✅ Customer: {fields['customer_name']}")

        # Product
        product_match = re.search(r'[Pp]roduct[:\s]+([A-Z][A-Za-z0-9\s]+?)(?=\s+[Mm]anufacturer|$)', text_clean)
        if product_match:
            fields["product_name"] = product_match.group(1).strip()
            print(f"      ✅ Product: {fields['product_name']}")

        return fields

# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("🚀 Initializing Warranty Validation System")
print("="*70)

ml_classifier = WarrantyClassifier(model_path="models/best_model_pytorch.pth")
ocr_engine = OCREngine()

print("✅ All systems ready!")
print("="*70 + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# TEST DATABASE WITH PATH VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

print("="*70)
print("🗄️  Testing Database Connection")
print("="*70)

try:
    # Show database file info
    print(f"\n📂 Database File:")
    print(f"   Path: {DB_PATH.resolve()}")
    print(f"   Exists: {DB_PATH.exists()}")
    if DB_PATH.exists():
        size = DB_PATH.stat().st_size
        print(f"   Size: {size:,} bytes")
    
    test_db = SessionLocal()
    
    customer_count = test_db.query(Customer).count()
    product_count = test_db.query(Product).count()
    warranty_count = test_db.query(Warranty).count()
    
    print(f"\n✅ Database connected!")
    print(f"   Customers: {customer_count}")
    print(f"   Products: {product_count}")
    print(f"   Warranties: {warranty_count}")
    
    if warranty_count > 0:
        print(f"\n   All Warranty Numbers in Database:")
        all_warranties = test_db.query(Warranty).all()
        for w in all_warranties:
            is_expired = w.expiry_date < datetime.now()
            status = "EXPIRED" if is_expired else "ACTIVE"
            print(f"      - '{w.warranty_number}' (ID:{w.id}, {status}, expires: {w.expiry_date.strftime('%d/%m/%Y')})")
        
        # Specifically check for WTYOCN37223690
        print(f"\n   🔍 Testing specific warranty lookup:")
        test_warranty = test_db.query(Warranty).filter(Warranty.warranty_number == "WTYOCN37223690").first()
        if test_warranty:
            print(f"      ✅ WTYOCN37223690 FOUND! (ID: {test_warranty.id})")
        else:
            print(f"      ❌ WTYOCN37223690 NOT FOUND")
            
        test_warranty2 = test_db.query(Warranty).filter(Warranty.warranty_number == "WTYLEB86074021").first()
        if test_warranty2:
            print(f"      ✅ WTYLEB86074021 FOUND! (ID: {test_warranty2.id})")
        else:
            print(f"      ❌ WTYLEB86074021 NOT FOUND")
    else:
        print(f"\n   ⚠️  Database is EMPTY!")
        print(f"   No warranty records found!")
    
    test_db.close()
except Exception as e:
    print(f"❌ Database error: {e}")
    import traceback
    traceback.print_exc()
    
print("="*70 + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Warranty Validation API - Final v6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {
        "service": "Warranty Validation API",
        "version": "6.0 - Final with Database Path Verification",
        "database_path": str(DB_PATH.resolve()),
        "database_exists": DB_PATH.exists(),
        "status": "running"
    }

@app.post("/api/v1/warranty/validate")
async def validate_warranty(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Complete warranty validation with database path verification"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / f"warranty_{timestamp}_{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"\n{'='*70}")
    print(f"📸 Processing: {file.filename}")
    print(f"{'='*70}")

    # STEP 1: ML Classification
    try:
        print(f"\n🤖 ML Classification")
        result = ml_classifier.predict(str(file_path))
        ml_pred = result['prediction']
        ml_conf = result['confidence']
        ml_probs = result['probabilities']
        is_authentic_ml = result['is_authentic']
        print(f"   {ml_pred.upper()} ({ml_conf*100:.1f}%)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # STEP 2: OCR
    print(f"\n📝 OCR Extraction")
    text = ocr_engine.extract_text(str(file_path))
    print(f"   Extracted {len(text)} characters")

    # STEP 3: Field Extraction
    print(f"\n🔍 Field Extraction")
    extracted = FieldExtractor.extract_all(text)
    warranty_number = extracted.get("warranty_number")

    # STEP 4: Database Check (ENHANCED WITH DEBUGGING)
    print(f"\n🗄️  Database Verification")
    database_match = False
    is_active = False
    db_status = "unknown"
    expiry_date_str = None
    
    if warranty_number:
        print(f"   Searching for: '{warranty_number}'")
        
        # Count total records
        total = db.query(Warranty).count()
        print(f"   Total in database: {total}")
        
        # Query
        record = db.query(Warranty).filter(Warranty.warranty_number == warranty_number).first()
        
        if record:
            database_match = True
            today = datetime.now()
            is_expired = record.expiry_date < today
            is_active = not is_expired
            db_status = "expired" if is_expired else "active"
            expiry_date_str = record.expiry_date.strftime('%d/%m/%Y')
            
            print(f"   ✅ FOUND: {warranty_number}")
            print(f"      ID: {record.id}")
            print(f"      Expiry: {expiry_date_str}")
            print(f"      Status: {db_status.upper()}")
        else:
            print(f"   ❌ NOT FOUND in database")
            print(f"   🔍 Checking all warranties for partial match...")
            all_warr = db.query(Warranty).all()
            for w in all_warr[:5]:
                print(f"      DB has: '{w.warranty_number}'")
    else:
        print(f"   ⚠️  No warranty number extracted")

    # STEP 5: Final Verdict
    print(f"\n⚖️  Final Verdict")
    
    if not warranty_number:
        verdict = "SUSPICIOUS"
        score = 0.5
        msg = "⚠️ Warranty number not found"
    elif not is_authentic_ml:
        verdict = "FRAUDULENT"
        score = ml_conf
        msg = "❌ Detected as fraudulent"
    elif database_match and is_active:
        verdict = "AUTHENTIC"
        score = 0.1
        msg = "✅ Valid warranty card"
    elif database_match and not is_active:
        verdict = "EXPIRED"
        score = 0.3
        msg = f"⚠️ Warranty expired on {expiry_date_str}"
    else:
        verdict = "SUSPICIOUS"
        score = 0.6
        msg = f"⚠️ Warranty {warranty_number} not in database"

    print(f"   {verdict}: {msg}")
    print(f"{'='*70}\n")

    # Cleanup
    try:
        file_path.unlink()
    except:
        pass

    return {
        "validation_id": f"VAL_{timestamp}",
        "timestamp": datetime.now().isoformat(),
        "filename": file.filename,
        "ml_classification": {
            "prediction": ml_pred,
            "confidence": float(ml_conf),
            "probabilities": {
                "authentic": float(ml_probs['authentic']),
                "fraudulent": float(ml_probs['fraudulent']),
            }
        },
        "extracted_data": extracted,
        "database_verification": {
            "found": database_match,
            "warranty_number": warranty_number,
            "status": db_status if database_match else None,
            "is_active": is_active if database_match else None,
            "expiry_date": expiry_date_str if database_match else None
        },
        "final_verdict": verdict,
        "fraud_risk_score": float(score),
        "message": msg
    }

if __name__ == "__main__":
    print("🌐 Starting Warranty Validation API")
    print("📍 URL: http://localhost:8080")
    print("📚 Docs: http://localhost:8080/docs")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8080)
