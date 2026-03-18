import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'c' to capture and OCR
    if key == ord('c'):
        h, w, _ = frame.shape
        # Crop top ~10% of the frame, center horizontally
        title_region = frame[0:int(h*0.1), int(w*0.2):int(w*0.8)]

        gray = cv2.cvtColor(title_region, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

        # OCR with single-line mode
        text = pytesseract.image_to_string(thresh, config="--psm 7")
        print("OCR output:", text)

        # Optional: show the cropped region for debugging
        cv2.imshow("Title Region", thresh)

    # Press 'v' to pause OCR (does nothing, just a message)
    elif key == ord('v'):
        print("Paused. Press 'c' to capture again.")

    # Press 'q' to exit completely
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
