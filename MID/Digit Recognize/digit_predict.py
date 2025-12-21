import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model("digit_classifier.h5")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        digit = thresh[y:y+h, x:x+w]
        digit = cv2.resize(digit, (28, 28))
        digit = digit.astype("float32") / 255.0
        digit = digit.reshape(1, 28 * 28)

        prediction = model.predict(digit)
        digit_class = np.argmax(prediction)
        confidence = prediction[0][digit_class]

        cv2.putText(frame, f"{digit_class} ({confidence*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.putText(frame, f"{digit_class} ({confidence*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        final_prediction = f"Final prediction: {digit_class} (Confidence: {confidence*100:.2f}%)"
        print(final_prediction)

        break

cap.release()
cv2.destroyAllWindows()
