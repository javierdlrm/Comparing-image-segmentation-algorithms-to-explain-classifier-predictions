import requests as r
from io import BytesIO
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np
from math import floor
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import tensorflow as tf
print("Tensorflow", tf.__version__)
from tensorflow.keras.applications import inception_v3 as inc_net

from tensorflow import keras
print("Keras", keras.__version__)
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image

import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm, BaseWrapper

from sklearn.metrics import confusion_matrix 
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries, inverse_gaussian_gradient, morphological_geodesic_active_contour
from skimage.util import img_as_float
from skimage.feature import canny
from skimage.morphology import remove_small_objects
from skimage.measure import label

import scipy.ndimage as ndi

from segmentation import CustomSegmentationAlgorithm


class Explanator:
    def __init__(
        self,
        inet_model=None,
        explainer=None,
        root=None,
        class_mapping=None,
        imagenet_classes=None,
    ):

        self.inet_model = inet_model if inet_model else inc_net.InceptionV3()
        self.explainer = explainer if explainer else lime_image.LimeImageExplainer()
        self.root = (
            root
            if root
            else ET.fromstring(
                r.get("http://www.image-net.org/api/xml/structure_released.xml").text
            )
        )
        self.imagenet_classes = (
            imagenet_classes
            if imagenet_classes
            else eval(
                r.get(
                    "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
                ).text
            )
        )

        if class_mapping:
            self.class_mapping = class_mapping
        else:
            self.class_mapping = {
                "person": {"wnid": "n00007846"},
                "bicycle": {"wnid": "n02834778"},
                "car": {"wnid": "n02958343"},
                "cat": {"wnid": "n02121808"},
            }

        self.wnid_mapping = {}
        self.fetch_mappings()

    def get_synset(self, wnid):
        for i in self.root[1].iter("synset"):
            if i.get("wnid") == wnid:
                return i

    def fetch_mappings(self):
        for i, v in self.class_mapping.items():
            # load sub_wnids
            synset = self.get_synset(v["wnid"])
            self.class_mapping[i]["sub_wnids"] = self.get_sub_wnids(synset)
            
            # load image net indices
            self.class_mapping[i]["imagenet_indices"] = []
            for ix, vx in self.imagenet_classes.items():
                if vx[0] in self.class_mapping[i]["sub_wnids"]:
                    self.class_mapping[i]["imagenet_indices"].append(ix)
            
            # load wnid_mapping
            for wnid in self.class_mapping[i]["sub_wnids"] + [v["wnid"]]:
                self.wnid_mapping[wnid] = i
                
    def download_and_preprocess(self, url):
        response = r.get(url)
        im = Image.open(BytesIO(response.content))
        im = im.resize((299, 299))
        x = image.img_to_array(im)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        return im, x

    def predict(self, x, raw=False):
        preds = self.inet_model.predict(x)
        if not raw:
            preds = decode_predictions(preds)[0]
        return preds

    def get_segmentation(self, anns, img_md):
        scaler_v = np.vectorize(self.scaler, excluded=["S", "cx"])

        polygons = []
        segs = []
        color = []
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]

        for seg in anns[0]["segmentation"]:
            poly, rescaled_seg = self.rescale_poly(
                seg, scaler_v, img_md["width"], img_md["height"]
            )
            polygons.append(poly)
            segs.append(rescaled_seg)
            color.append(c)
        return color, polygons, segs

    def explain(self, img, img_md, segmentation_fns, coco_obj, top_labels=5, num_samples=1000):
        annIds = coco_obj.getAnnIds(imgIds=img_md["id"], catIds=[], iscrowd=None)
        anns = coco_obj.loadAnns(annIds)
        color, polygons, segs = self.get_segmentation(anns, img_md)
        img_shape = (img.shape[0], img.shape[1])

        # poly mask: ground-truth
        img_new = Image.new("L", img_shape, 0)
        ImageDraw.Draw(img_new).polygon(segs[0].tolist(), outline=1, fill=1)
        poly_mask = np.array(img_new)

        output = {}
        for segmentation_fn in segmentation_fns:
            print("Start explain_instance")
            explanation = self.explainer.explain_instance(
                img,
                self.inet_model.predict,
                top_labels=top_labels,
                hide_color=0,
                num_samples=num_samples,
                segmentation_fn=CustomSegmentationAlgorithm(segmentation_fn),
            )
            print("Finish explain_instance")
            union_mask = np.zeros(img_shape, dtype=int)
            output[segmentation_fn] = {}
            
            relevant_indices = self.class_mapping["cat"]["imagenet_indices"]
            relevant_indices = [int(i) for i in relevant_indices]
            for i in relevant_indices:
                temp, mask = explanation.get_image_and_mask(
                    i, positive_only=True, hide_rest=True
                )
                classes = self.imagenet_classes[str(i)]
                union_mask = (
                    mask if union_mask is None else np.bitwise_or(union_mask, mask)
                )

                iou, dice, pixel_acc = self.evaluate_segmentation(mask, poly_mask)
                print(f"Labels {self.imagenet_classes[str(i)]} evaluation")
                print(f"IOU: {iou}")
                print(f"DICE: {dice}")
                print(f"Pixel acc: {pixel_acc}")

                # to-do: Subplots
                self.plot_and_save_segmentation(
                    img_md["id"], img, mask, polygons, color, segmentation_fn, classes
                )
                output[segmentation_fn][str(classes)] = {
                    "mask": mask,
                    "iou": iou,
                    "dice": dice,
                    "pixel_acc": pixel_acc,
                }

            # evaluate union masks per segmentation
            iou, dice, pixel_acc = self.evaluate_segmentation(union_mask, poly_mask)
            print(f"Segmentation {segmentation_fn} union mask evaluation:")
            print(f"IOU: {iou}")
            print(f"DICE: {dice}")
            print(f"Pixel acc: {pixel_acc}")

            joined_classes = "-".join(
                np.concatenate(
                    [self.imagenet_classes[str(i)] for i in relevant_indices],
                    axis=0,
                )
            )
            self.plot_and_save_segmentation(
                img_md["id"], img, union_mask, polygons, color, segmentation_fn, joined_classes
            )
            output[segmentation_fn]["union"] = {
                "mask": union_mask,
                "iou": iou,
                "dice": dice,
                "pixel_acc": pixel_acc,
            }

        return output

    def plot_and_save_segmentation(
        self, img_id, img, mask, polygons, color, segmentation_fn, classes
    ):
        plt.figure(figsize=(10, 10))
        plt.imshow(mark_boundaries(img / 2 + 0.5, mask))
        plt.title(segmentation_fn + " / " + str(classes))

        ax = plt.gca()
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor="none", edgecolors=color, linewidths=2)
        ax.add_collection(p)
        plt.axis("off")
        plt.show()
        if not os.path.exists(os.path.join("..", "figures", str(img_id))):
            os.makedirs(os.path.join("..", "figures", str(img_id)))
        plt.savefig(os.path.join("..", "figures", str(img_id), segmentation_fn + "_" + classes[1] + ".png"))

    @staticmethod
    def scaler(x, S, cx):
        return (S * (x - cx)) + cx

    @staticmethod
    def rescale_poly(poly, scaler_v, width, height):
        # poly: [x1,y1,x2,y2,...]
        poly = np.array(poly).reshape((int(len(poly) / 2), 2))
        poly[:, 0] = scaler_v(poly[:, 0], S=299 / width, cx=0)
        poly[:, 1] = scaler_v(poly[:, 1], S=299 / height, cx=0)
        return Polygon(poly), poly.flatten()

    @staticmethod
    def evaluate_segmentation(y_pred, y_true):
        # essential metrics: Pixel accuracy, Intersection-Over-Union (Jaccard index) and Dice coeff (f1 score)
        # intro: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2

        assert y_pred.shape == y_true.shape, "Input masks should be same shape"

        vec_y_pred = y_pred.flatten()
        vec_y_true = y_true.flatten()

        # confusion matrix
        current = confusion_matrix(vec_y_true, vec_y_pred, labels=[0, 1])
        tn, fp, fn, tp = current.ravel()

        # metrics
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
        pixel_acc = (fp + fn) / (tp + tn + fp + fn)

        # IOU alternatives

        # using coco_mask: NOT WORKING
        # patch -> asfortranarray: https://github.com/cocodataset/cocoapi/issues/91
        #     inter = np.bitwise_and(vec_y_pred, vec_y_true)
        #     enc_inter = coco_mask.encode(inter)
        #     enc_union = coco_mask.encode(np.bitwise_or(vec_y_pred, vec_y_true))
        #     iou_coco = coco_mask.area(enc_inter) / coco_mask.area(enc_union)
        #     iou_coco = coco_mask.iou(enc_y_pred, enc_y_true, 0)
        #     print(f"IOU COCO: {iou_coco}")

        # using tf.metrics.mean_iou: NOT WORKING
        #     with tf.Session() as sess:
        #       ypredT = tf.constant(np.argmax(vec_y_pred, axis=-1))
        #       ytrueT = tf.constant(np.argmax(vec_y_true, axis=-1))
        #       iou, conf_mat = tf.metrics.mean_iou(ytrueT, ypredT, num_classes=2)
        #       sess.run(tf.local_variables_initializer())
        #       sess.run([conf_mat])
        #       miou = sess.run(iou)
        #       print(f"MIOU: {miou}")

        # using tf.keras.metrics.MeanIoU: NOT WORKING
        #     with tf.Session() as sess:
        #       m = tf.keras.metrics.MeanIoU(num_classes=2)
        #       m.update_state(vec_y_true, vec_y_pred)
        #       res_tensor = m.result()
        #       sess.run(tf.local_variables_initializer())
        #       iou_keras = sess.run(res_tensor)
        #       print(f"IOU KERAS: {iou_keras}")

        return iou, dice, pixel_acc

    @staticmethod
    def get_sub_wnids(synset):
        sub_wnids = []
        for i in synset.iter("synset"):
            sub_wnids.append(i.get("wnid"))
        return sub_wnids

    def plot_segmentations(self, img, segmentation_fns):
        n_cols = 2
        n_rows = floor(len(segmentation_fns) / n_cols)
        if len(segmentation_fns) % n_cols > 0:
            n_rows += 1

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 10), sharex=True, sharey=True)
        cnt = -1

        for segmentation_fn, params in segmentation_fns.items():
            cnt += 1

            # eval segmentation
            if segmentation_fn == 'canny':
                img_gray = rgb2gray(img)
                edges = canny(img_gray, sigma=params['sigma'])
                fill_edges = ndi.binary_fill_holes(edges)
                segments = ndi.label(remove_small_objects(fill_edges, params['min_size']))[0]
                print(f"{segmentation_fn}(img, sigma={params['sigma']}) - min_size=params['min_size']")
            elif segmentation_fn == 'morphological_geodesic_active_contour':
                img_gray = rgb2gray(img)
                img_float = img_as_float(img_gray)
                gradient = inverse_gaussian_gradient(img_float)
                init_ls = np.zeros(img_gray.shape, dtype=np.int8)
                init_ls[10:-10, 10:-10] = 1
                params['init_ls'] = init_ls
                raw_segments = morphological_geodesic_active_contour(gradient, iterations=params['iterations'],
                                init_level_set=init_ls, smoothing=params['smoothing'], balloon=params['balloon'],
                                threshold=params['threshold'])
                segments = label(raw_segments)
            else:
                segments = self.get_segments(img, params, segmentation_fn)
        
            print(f"{segmentation_fn} number of segments: {len(np.unique(segments))}")

            # plot
            r = floor(cnt / n_cols)
            c = cnt % n_cols
            ax[r, c].imshow(mark_boundaries(img, segments))
            ax[r, c].set_title(segmentation_fn)

        for a in ax.ravel():
            a.set_axis_off()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_segments(img, params, segmentation_fn):
        img_param = img if segmentation_fn != "watershed" else sobel(rgb2gray(img))
        str_params = ", ".join([f"{k}={v}" for k, v in params.items()])
        function = segmentation_fn + "(img_param, " + str_params + ")"
        print(function)
        segments = eval(function)
        return segments
