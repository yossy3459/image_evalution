"""
2つの画像を比較し、MSE, SSIM, PSNRを算出する.
"""
import math
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2


def mse(image_a, image_b):
    """
    2つの画像の平均二乗誤差を求める (MSE)
    @param image_a 画像1
    @param image_b 画像2
    """
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    # Translate: 2つの画像間の'MSE'は、2つの画像の要素の差を2乗したものの平均です。
    #            注意: 2つの画像は同じ次元でなければなりません
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    # Translate: MSE値を返す、MSEはエラー値が低いほど2つの画像がより'似ている'
    return err

def psnr(mse_value):
    """
    mse値からpsnr値を算出する.  自作関数
    @param mse_value mse値
    """
    if mse_value == 0:
        ans = None
    else:
        ans = 20 * math.log10(255 / math.sqrt(mse_value))
    return ans

def compare_images(image_a, image_b, title):
    """
    2つの画像を比較する関数.
    @param image_a 画像A
    @param image_b 画像B
    @param title 出力画像につけるタイトル
    """

    # compute the mean squared error and structural similarity
    # index for the images
    # Translate: 平均二乗誤差と構造的類似度の計算をする
    result_mse = mse(image_a, image_b)
    result_psnr = psnr(result_mse)
    result_ssim = ssim(image_a, image_b)

    # setup the figure
    # Translate: 図の設定
    fig_compare = plt.figure(title)
    if result_psnr == None:
        plt.suptitle("MSE: %.4f, PSNR: undefined, SSIM: %.4f" % (result_mse, result_ssim))
    else:
        plt.suptitle("MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (result_mse, result_psnr, result_ssim))

    # show first image
    # Translate: 1枚目の画像を表示
    fig_compare.add_subplot(1, 2, 1)
    plt.imshow(image_a)
    plt.gray()
    plt.axis("off")

    # show the second image
    # translate: 2枚目の画像を表示
    fig_compare.add_subplot(1, 2, 2)
    plt.imshow(image_b)
    plt.gray()
    plt.axis("off")

    # show the images
	# Translate: 画像を表示
    plt.show()

# load the images -- the original, the original + proposed,
# and the original + intel
# Translate: 画像を読み込む - オリジナル、オリジナルと提案手法、オリジナルとインテルの手法
# cv2.IMREAD_GRAYSCALEにより、最初から2値画像を読み込むこととする
original = cv2.imread("img/image_without_AC.tif", cv2.IMREAD_GRAYSCALE)
proposed = cv2.imread("img/image_proposedAC_r_tolerance_20_cache_50.tif", cv2.IMREAD_GRAYSCALE)
intel = cv2.imread("img/image_intelAC_tolerance_20_cache_50.tif", cv2.IMREAD_GRAYSCALE)

plt.imshow(original)
plt.gray()

# convert the images to grayscale
# Translate: 2値画像へ変換
# 不要なためコメントアウト
# original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
# proposed = cv2.cvtColor(proposed, cv2.COLOR_BGR2GRAY)
# intel = cv2.cvtColor(intel, cv2.COLOR_BGR2GRAY)

# initialize the figure
# Translate: 図の初期化
fig = plt.figure("Images")
images = ("Original", original), ("Proposed", proposed), ("iACT", intel)

# loop over the images
# Translate: 各画像に対してループ処理
for (i, (name, image)) in enumerate(images):
    # show the image
    # Translate: 画像の表示
    axis = fig.add_subplot(1, 3, i + 1)
    axis.set_title(name)
    plt.imshow(image)
    plt.gray()
    plt.axis("off")

# show the figure
# Translate: 図の表示
plt.show()


# compare the images
# Translate: 画像の比較
compare_images(original, original, "Original vs. Original")
compare_images(original, proposed, "Original vs. ProposedAC")
compare_images(original, intel, "Original vs. IntelAC")
