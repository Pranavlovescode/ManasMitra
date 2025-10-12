#!/usr/bin/env python3
"""
Test script to verify the fixes in suiciderisk_UI.py
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the functions from the main file
from suiciderisk_UI import translate_to_english, predict_text

def test_translation_function():
    """Test the translate_to_english function with various inputs"""
    print("Testing translate_to_english function...")
    
    # Test with English text
    result = translate_to_english("Hello, this is a test message.")
    print(f"English text test: {result}")
    
    # Test with empty string
    result = translate_to_english("")
    print(f"Empty string test: {result}")
    
    # Test with short text
    result = translate_to_english("Hi")
    print(f"Short text test: {result}")
    
    print("Translation function tests completed.\n")

def test_prediction_function():
    """Test the predict_text function with various inputs"""
    print("Testing predict_text function...")
    
    # Test with normal text
    result = predict_text("I am feeling great today!")
    print(f"Positive text test: {result}")
    
    # Test with empty string
    result = predict_text("")
    print(f"Empty string test: {result}")
    
    # Test with very short text
    result = predict_text("Hi")
    print(f"Short text test: {result}")
    
    print("Prediction function tests completed.\n")

if __name__ == "__main__":
    print("Starting error handling tests...\n")
    
    try:
        test_translation_function()
        test_prediction_function()
        print("All tests completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")