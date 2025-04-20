import cv2
import requests

# URL de ton API Flask
API_URL = "http://localhost:8080/predict"

# Capture depuis la webcam
cap = cv2.VideoCapture(0)
print("Appuyez sur ESPACE pour capturer une image")

while True:
    ret, frame = cap.read()
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key % 256 == 32:  # Espace pour capturer
        # Sauvegarder temporairement l'image
        img_path = "captured_image.jpg"
        cv2.imwrite(img_path, frame)
        print(f"[INFO] Image capturée et enregistrée sous {img_path}")
        break

cap.release()
cv2.destroyAllWindows()

# Envoi à l'API Flask
with open(img_path, "rb") as img_file:
    files = {"image": img_file}
    response = requests.post(API_URL, files=files)

# Résultat
try:
    result = response.json()
    print("Résultat de la prédiction :", result)
except Exception as e:
    print("Erreur lors de la lecture de la réponse :", e)
    print("Texte brut :", response.text)
