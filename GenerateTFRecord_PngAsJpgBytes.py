# Demonstration of creating TFRecord file with images stored as bytes and PNG images converted to JPEG

import tensorflow as tf
import os
import matplotlib.image as mpimg

class GenerateTFRecord:
    def __init__(self, labels):
        self.labels = labels
        self._create_graph()

    def convert_image_folder(self, img_folder, tfrecord_file_name):
        # Get all file names of images present in folder
        img_paths = os.listdir(img_folder)
        img_paths = [os.path.abspath(os.path.join(img_folder, i)) for i in img_paths]

        with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer, tf.Session() as sess:
            for img_path in img_paths:
                example = self._convert_image(img_path)
                writer.write(example.SerializeToString())

    # Create graph to convert PNG image data to JPEG data
    def _create_graph(self):
        tf.reset_default_graph()
        self.png_img_pl = tf.placeholder(tf.string)
        png_enc = tf.image.decode_png(self.png_img_pl, channels = 3)
        # Set how much quality of image you would like to retain while conversion
        self.png_to_jpeg = tf.image.encode_jpeg(png_enc, format = 'rgb', quality = 100)

    def _is_png_image(self, filename):
        ext = os.path.splitext(filename)[1].lower()
        return ext == '.png'

    # Run graph to convert PNG image data to JPEG data
    def _convert_png_to_jpeg(self, img):
        sess = tf.get_default_session()
        return sess.run(self.png_to_jpeg, feed_dict = {self.png_img_pl: img})

    def _convert_image(self, img_path):
        label = self._get_label_with_filename(img_path)
        img_shape = mpimg.imread(img_path).shape
        filename = os.path.basename(img_path).split('.')[0]

        # Read image data in terms of bytes
        with tf.gfile.FastGFile(img_path, 'rb') as fid:
            image_data = fid.read()

            # Encode PNG data to JPEG data
            if self._is_png_image(img_path):
                image_data = self._convert_png_to_jpeg(image_data)

        example = tf.train.Example(features = tf.train.Features(feature = {
            'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),
            'rows': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[0]])),
            'cols': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[1]])),
            'channels': tf.train.Feature(int64_list = tf.train.Int64List(value = [3])),
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data])),
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
        }))
        return example

    def _get_label_with_filename(self, filename):
        basename = os.path.basename(filename).split('.')[0]
        basename = basename.split('_')[0]
        return self.labels[basename]

if __name__ == '__main__':
    labels = {'cat': 0, 'dog': 1}
    t = GenerateTFRecord(labels)
    t.convert_image_folder('Images', 'images3.tfrecord')
