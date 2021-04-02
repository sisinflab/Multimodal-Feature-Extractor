import numpy as np
import cv2


class LowFeatureExtractor:
    def __init__(self, args):
        self.num_bins = args.num_bins

    def extract_color_shape(self, sample):

        image, name = sample

        # shape extraction
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Ie1 = cv2.Canny(gray, 255 / 3, 255)
        f = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        Ie2 = cv2.filter2D(gray, -1, f)
        Ie = Ie1 + Ie2
        Ie_end = np.clip(255 - Ie, a_min=0, a_max=255)

        # color histogram extraction
        contours, hierarchy = cv2.findContours(Ie, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_info = []
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))

        try:
            max_contour = sorted(contour_info, key=lambda cont: cont[2], reverse=True)[0]
            mask = np.copy(image)
            cv2.fillPoly(mask, pts=[max_contour[0]], color=(0, 0, 0))
            image_for_histogram = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            temp = (mask == 0).all(axis=2).astype(np.uint8)
            hist = cv2.calcHist([image_for_histogram], [0, 1, 2], temp, [self.num_bins]*3, [0, 255, 0, 255, 0, 255])
            hist = np.asarray(hist, dtype=np.int32).flatten()
        except IndexError:
            print(f'''\n\r****{name} shape cannot be extracted properly!****''')
            hist = np.zeros((1, self.num_bins * self.num_bins * self.num_bins))

        return hist, Ie_end
