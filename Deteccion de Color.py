import cv2 as cv
import numpy as np

video = cv.VideoCapture(0)

while(True):
    ret, frame = video.read()
    imagenOriginal = frame
    imagenWindows = cv.imread("Hola Gato.jpeg")

    # filas,columnas,canales=imagenOriginal.shape
    # filas2,columnas2,canales2=np.shape(imagenWindows)

    imagenWindows = cv.resize(imagenWindows, (540,540), interpolation=cv.INTER_NEAREST)
    imagenOriginal = cv.resize(imagenOriginal, (540,540), interpolation=cv.INTER_NEAREST)

    minBGR = np.array([0, 100, 0])
    maxBGR = np.array([150, 255, 150])

    maskBGR = cv.inRange(imagenOriginal, minBGR, maxBGR)
    mask_inv = cv.bitwise_not(maskBGR)

    resultBGR = cv.bitwise_and(imagenOriginal, imagenOriginal, mask=mask_inv)
    result_inv = cv.bitwise_and(imagenWindows, imagenWindows, mask=maskBGR)

    total = cv.add(resultBGR, result_inv)
    resize = cv.resize(total, (1280, 940), interpolation=cv.INTER_NEAREST)
    cv.imshow("Resultado Final", resize)

    if cv.waitKey(1) == ord('q'):
        break

video.release()
cv.destroyAllWindows()