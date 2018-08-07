# Demonstration to add and extract images from TFRecord with Dataset and Iterator
Illustration of how to create TFRecord with images and pipelined into Datasets and Iterators.

You can read the full detailed explanation about the code used in this repository in [my Medium post]
(https://medium.com/ymedialabs-innovation/how-to-use-tfrecord-with-datasets-and-iterators-in-tensorflow-with-code-samples-ffee57d298af).

You may go through each of the file for the specific code samples listed below.

## Generation of TFRecords
Code sample showing how to create TFRecord with an array of text-based dictionaries.
```
python GenerateSimpleTFRecord.py
```

Code sample showing how to create TFRecord with images stored as Numpy string.
```
python GenerateTFRecord_ImgAsStr.py
```

Code sample showing how to create TFRecord with image files stored as Bytes.
```
python GenerateTFRecord_ImgAsBytes.py
```

Code sample showing how to create TFRecord with image files stored as Bytes along with PNG image files converted to JPEG.
```
python GenerateTFRecord_PngAsJpgBytes.py
```


## Extraction of TFRecords
Code sample showing how to extract TFRecord consisting of array of text-based dictionaries by TFRecordDataset and Iterator.
```
python ExtractSimpleTFRecord.py
```

Code sample showing how to extract TFRecord consisting of images stored as Bytes by TFRecordDataset and Iterator.
```
python ExtractTFRecord_ImgAsBytes.py
```

Code sample showing how to extract TFRecord consisting of images stored as Bytes along with PNG image files converted to JPEG by TFRecordDataset and Iterator.
```
python ExtractTFRecord_PngAsJpgBytes.py
```
