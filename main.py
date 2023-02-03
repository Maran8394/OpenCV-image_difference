import cv2
import imutils
import numpy as np
import uuid


class ImageDifference():
    def __init__(self,img1, img2) -> None:
        self.IMG_SIZE = (400, 500)
        self.img1 = cv2.imread(img1)
        self.img2 = cv2.imread(img2)

        if self.img1.shape != self.img2.shape:
            self.img1 = cv2.resize(self.img1, self.IMG_SIZE)
            self.img2 = cv2.resize(self.img2, self.IMG_SIZE)

    def denoising_and_converting_to_gray(self):
        dst1 = cv2.fastNlMeansDenoisingColored(self.img1,None,10,10,7,21)
        dst2 = cv2.fastNlMeansDenoisingColored(self.img2,None,10,10,7,21)
        gray1 = cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)
        return [gray1, gray2]
    

    def erosionAndDilation(self,thresh):
        # FIXME needs to calculate erosion and dilation kernal value based on the pic threshold 
        eros_kernal = np.ones((3,3),np.uint8)
        eros = cv2.erode(thresh,eros_kernal,iterations=3)
        dil_kernal = np.ones((3,3),np.uint8)
        dil = cv2.dilate(eros,dil_kernal,iterations=6)
        return dil

    def get_contours(self,image):
        # contours
        contours = cv2.findContours(image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        return contours

    def hightlight_difference(self, contours,highlight=True, save=True):
        uid = uuid.uuid4()
        CROPPED = None
        for contour in contours:
            if cv2.contourArea(contour)>100:
                (x, y, w, h) = cv2.boundingRect(contour)
                cropped = self.img2[y: y+h, x: x+w]
                CROPPED = cropped
                if highlight:
                    # cv2.rectangle(reA, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.rectangle(self.img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
                if save:
                    img_name = f"cropped/{uid}.png"
                    cv2.imwrite(img_name, cropped)
        return CROPPED
        

    def findDifferenceAndThreshold(self):
        gray1, gray2 = self.denoising_and_converting_to_gray()
        difference = cv2.subtract(gray1, gray2)
        thresh = cv2.threshold(difference, 60, 255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
        dilated_image = self.erosionAndDilation(thresh=thresh)
        contours = self.get_contours(image=dilated_image)
        cutted_pixels = self.hightlight_difference(contours=contours,highlight=True,save=True)
        cv2.imshow("result", self.img2)
        cv2.imshow("cutted_pixels", cutted_pixels)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    i = ImageDifference(
        img1="projectImages/withoutItem.jpeg",
        img2="projectImages/withItem.jpeg",
    ).findDifferenceAndThreshold()
