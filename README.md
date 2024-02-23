<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png?3"
      >
    </a>
  </p>
</div>

# Autodistill YOLOv5 Module

This repository contains the code supporting the YOLOv5 target model for use with [Autodistill](https://github.com/autodistill/autodistill).

[YOLOv5](https://github.com/ultralytics/ultralytics) is an open-source computer vision model by Ultralytics, the creators of YOLOv5. You can use `autodistill` to train a YOLOv5 object detection model on a dataset of labelled images generated by the base models that `autodistill` supports.

View our [YOLOv5 Instance Segmentation](/target-models/YOLOv5-instance-segmentation/) page for information on how to train instance segmentation models.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [YOLOv5 Autodistill documentation](https://autodistill.github.io/autodistill/target_models/yolov5/).

## Installation

To use the YOLOv5 target model, you will need to install the following dependency:

```bash
pip3 install autodistill-yolov5
```

## Quickstart

```python
from autodistill_yolov5 import YOLOv5

target_model = YOLOv5("YOLOv5n")

# train a model
target_model.train("./context_images_labeled/data.yaml", epochs=200)

# run inference on the new model
pred = target_model.predict("./context_images_labeled/train/images/dog-7.jpg", conf=0.01)
```

## License

The code in this repository is licensed under an [AGPL 3.0 license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
