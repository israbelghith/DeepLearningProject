import cv2
import requests

# URL de l'API Flask
API_URL = "http://localhost:8080/predict"

# Ouvre la webcam
cap = cv2.VideoCapture(0)
print("📷 Appuyez sur ESPACE pour capturer une image...")

while True:
    ret, frame = cap.read()
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key % 256 == 32:  # ESPACE
        img_path = "captured_image.jpg"
        cv2.imwrite(img_path, frame)
        print(f"[✔] Image capturée : {img_path}")
        break

cap.release()
cv2.destroyAllWindows()

# Envoi de l'image à l'API
with open(img_path, "rb") as img_file:
    files = {"image": img_file}
    response = requests.post(API_URL, files=files)

# Affichage du résultat
try:
    result = response.json()
    print("🧠 Prédiction :", result)
except Exception as e:
    print("❌ Erreur :", e)
    print("Réponse brute :", response.text)
