"""
Simple test script to verify the crop recommendation system works correctly.
"""

from crop_recommendation import recommend_crop

print("=" * 60)
print("CROP RECOMMENDATION SYSTEM - QUICK TEST")
print("=" * 60)

# Test 1: Valid input for rice
print("\nTest 1: Valid Input - Rice")
result1 = recommend_crop(90, 42, 43, 20.8, 82, 6.5, 202.9)
print(f"Input: N=90, P=42, K=43, Temp=20.8, Humidity=82, pH=6.5, Rainfall=202.9")
print(f"Result: {result1}")

# Test 2: Invalid input - Temperature too high
print("\nTest 2: Invalid Input - Temperature Out of Range")
result2 = recommend_crop(90, 42, 43, 120, 82, 6.5, 202.9)
print(f"Input: N=90, P=42, K=43, Temp=120, Humidity=82, pH=6.5, Rainfall=202.9")
print(f"Result: {result2}")

# Test 3: Valid input for maize
print("\nTest 3: Valid Input - Maize")
result3 = recommend_crop(80, 43, 16, 23.5, 71.5, 6.6, 66.7)
print(f"Input: N=80, P=43, K=16, Temp=23.5, Humidity=71.5, pH=6.6, Rainfall=66.7")
print(f"Result: {result3}")

# Test 4: Invalid input - Humidity out of range
print("\nTest 4: Invalid Input - Humidity Out of Range")
result4 = recommend_crop(90, 42, 43, 25, 150, 6.5, 200)
print(f"Input: N=90, P=42, K=43, Temp=25, Humidity=150, pH=6.5, Rainfall=200")
print(f"Result: {result4}")

print("\n" + "=" * 60)
print("QUICK TEST COMPLETE")
print("=" * 60)
