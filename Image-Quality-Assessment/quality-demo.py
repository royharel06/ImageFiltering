from src.inference import get_image_quality_scores

result = get_image_quality_scores("B3_HZ_640x480_35.jpg")
print(result)

if result["score"] >= 70:
    print("Image passed quality threshold")
else:
    print("Image too blurry")
