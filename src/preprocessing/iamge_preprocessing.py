import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
from scipy.ndimage import interpolation as inter

class ImagePreprocessor:
    def __init__(self, folder_path= "../../eval_data/", real_image=True, image_name="page_00", image_type="png", image_show=False):
        """
        Initializes the preprocessor with the input image path.
        :param image_name: image name to the input document image.
        """
        self.real_image = real_image
        if self.real_image:
            self.image_name = image_name
            self.image_type = image_type
            self.image_show = image_show
            self.image_path = f"{folder_path}{image_name}.{image_type}"
            self.image = cv2.imread(self.image_path)
        self.gray = None
        self.denoised = None
        self.binary = None
        self.corrected = None
        self.result = None

    def convert_to_grayscale(self):
        """
            Converts the input image to grayscale.
            Input: BGR image
            Output: Grayscale image
        """
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.gray

    def denoise_image(self):
        """
            Applies median filtering for noise reduction.
            Input: Grayscale image
            Output: Denoised image
        """
        self.denoised = cv2.medianBlur(self.gray, 1)
        return self.denoised

    def binarize_with_sauvola(self):
        """
            Applies Sauvola thresholding to binarize the image.
            Input: Denoised image
            Output: Binary image
        """
        thresh_sauvola = threshold_sauvola(self.denoised, window_size=25)
        self.binary = (self.denoised > thresh_sauvola).astype(np.uint8) * 255
        return self.binary

    def correct_skew(self, delta=1, limit=5):
        """
            Detects and corrects document skew using Hough Transform.
            Skips correction if median line angle is close to 0°.
            Input: Binary image
            Output: Skew-corrected image
        """

        def determine_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1, dtype=float)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
            return histogram, score

        thresh = cv2.threshold(self.binary, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            histogram, score = determine_score(thresh, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]

        (h, w) = self.binary.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        self.corrected  = cv2.warpAffine(self.binary,
                                         M,
                                         (w, h),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)

        return self.corrected

    def apply_morphology(self):
        """
            Applies dilation followed by erosion (opening) to reinforce structures.
            Input: Skew-corrected image
            Output: Morphologically processed image
        """
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(self.corrected, kernel, iterations=1)
        self.result = cv2.erode(dilated, kernel, iterations=1)
        return self.result

    def show_image(self, title=""):
        """
            Utility to visualize and optionally save an image.
            :param img: Image to display
            :param title: Title for display and filename
            :param to_save: If True, saves the image
        """
        plt.figure(figsize=(12, 10))
        plt.imshow(self.result, cmap='gray')
        plt.axis("off")
        save_path = f"../../input/preprocessed/{title}{self.image_name}.{self.image_type}"
        plt.savefig(save_path)
        if self.image_show:
            plt.show()
        else:
            plt.close()

    def run_preprocessing(self):
        """
            Runs the full preprocessing step by step.
            Returns the final preprocessed result.
        """
        self.convert_to_grayscale()
        self.denoise_image()
        self.binarize_with_sauvola()
        self.correct_skew()
        self.apply_morphology()
        if self.real_image:
            self.show_image()
        return self.result

# Example usage:
# image_name = "page_04"
# preprocessor = ImagePreprocessor(image_name, image_show=True)
# processed_image = preprocessor.run_preprocessing()
