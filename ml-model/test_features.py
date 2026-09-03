import pytest
from train_model import extract_features

# =============================================
# TEST SUITE FOR FEATURE EXTRACTION
# =============================================

def test_extract_features_returns_five_values():
    """Ensure the function always returns exactly 5 features."""
    features = extract_features("Hello world")
    assert len(features) == 5, f"Expected 5 features, got {len(features)}"


def test_extract_features_with_high_urgency():
    """Test that urgency keywords and exclamations are detected."""
    text = "URGENT! System is down ASAP!!!"
    features = extract_features(text)
    
    # Feature 0: Length (30/100 = 0.3)
    assert features[0] == 0.3, f"Length feature mismatch: expected 0.3, got {features[0]}"
    
    # Feature 1: Exclamations (there are 4 '!' -> 4/5 = 0.8)
    assert features[1] == 0.8, f"Exclamation feature mismatch: expected 0.8, got {features[1]}"
    
    # Feature 2: Caps (11 caps -> 11/10 = 1.1 capped to 1.0)
    assert features[2] == 1.0, f"Caps feature mismatch: expected 1.0, got {features[2]}"
    
    # Feature 3: Urgency score (urgent + asap = 2)
    assert features[3] == 2.0, f"Urgency feature mismatch: expected 2.0, got {features[3]}"
    
    # Feature 4: Hour (should be between 0 and 1)
    assert 0 <= features[4] <= 1.0, f"Hour feature out of range: {features[4]}"


def test_extract_features_with_low_urgency():
    """Test a normal, polite message."""
    text = "Hello, I just wanted to ask about my account balance."
    features = extract_features(text)
    
    # Feature 0: Length > 0 (it's 52 characters)
    assert features[0] > 0, "Length should be > 0"
    
    # Feature 1: No exclamations
    assert features[1] == 0, f"No exclamations expected, got {features[1]}"
    
    # Feature 2: Caps: 'H' and 'I' → 2 caps → 2/10 = 0.2
    assert features[2] == 0.2, f"Caps feature mismatch: expected 0.2, got {features[2]}"
    
    # Feature 3: No urgency keywords
    assert features[3] == 0, f"No urgency keywords expected, got {features[3]}"
    
    # Feature 4: Hour (0-1)
    assert 0 <= features[4] <= 1.0, f"Hour feature out of range: {features[4]}"


def test_extract_features_caps_capping():
    """Test that caps count is capped at 1.0."""
    text = "AAAAA BBBBB CCCCC"  # 15 caps, should be capped at 1.0
    features = extract_features(text)
    assert features[2] == 1.0, f"Caps should be capped at 1.0, got {features[2]}"


def test_extract_features_handles_empty_string():
    """Test edge case: empty string."""
    features = extract_features("")
    assert features[0] == 0.0, "Empty string should have length 0"
    assert features[1] == 0.0
    assert features[2] == 0.0
    assert features[3] == 0.0