# Hector Vision

Contains

- hector_detection_aggregator
- hector_motion_detection
- hector_tag_detection
- hector_vision_test


## hector_detection_aggregator

Contains a node collecting data on detected objects and overlays them on an image feed.
Subscribed detections are buffered and synchronized based on their time stamps.
Detections tagged with the detection type and an additional value (e.g. the string of a qr-code).
Detections are colored according to their type:

| Detection Type | Color     |
| -------------- | --------- |
| motion         | red       |
| qr-code        | blue      |
| apriltag       | purple    |
| hazmat         | turquoise |

If avaivable the thermal image is also integrated into this visualization.

#### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `visualization_topic` | `string` | `"aggregated_detections"` | Topic where the visualizations of aggregated detections are published to. |
| `cam_topic` | `string` | `"camera/image_raw"` | Camera topic name all detections are extracted from. |
| `use_thermal` | `bool` | `false` | Whether to use thermal images for detection aggregation. |
| `thermal_topic` | `string` | `""` | Topic where thermal images are published. |
| `thermal_eps` | `int` | `1000000000` | Time tolerance in nanoseconds for approximate time synchronization with other input types. |
| `use_april` | `bool` | `false` | Whether to use april tag detections for detection aggregation. |
| `use_hazmat` | `bool` | `false` | Whether to use hazmat detections for detection aggregation. |
| `use_qr` | `bool` | `true` | Whether to use QR detections for detection aggregation. |
| `object_detection_topic` | `string` | `""` | Topic where object detections are published. This includes hazmats, apriltags, and QR codes. |
| `use_motion` | `bool` | `false` | Whether to use motion detections for detection aggregation. |
| `motion_topic` | `string` | `""` | Topic where motion detections are published. |
| `buffer_size` | `int` | `20` | Maximum amount of stored messages per detection type by time sync buffers. |
| `debug` | `bool` | `false` | Whether node is in debug mode. |

## hector_motion_detection

The motion_detection node can detect multiple moving objects in an image sequence.
                      |

#### Parameters

| Parameter                       | Type     | Default | Description                                   |
| ------------------------------- | -------- | ------- | --------------------------------------------- |
| `moving_average_weight`         | `double` | 1.0     | Weight of the new image                       |
| `activation_threshold`          | `int`    | 170     | Threshold for a pixel to be considered moving |
| `automatic_learning_rate`       | `bool`   | `false` | Automatic learning rate                       |
| `learning_rate`                 | `double` | 0.7     | Learning rate for background subtraction      |
| `motion_detect_detection_limit` | `int`    | 4       | Maximum number of motions to detect           |
| `motion_detect_min_area`        | `int`    | 60      | Minimal area of detected motions              |
| `motion_detect_max_area`        | `int`    | 5000    | Maximal area of detected motions              |
| `motion_detect_erosion`         | `int`    | 2       | Iterations for erosion on fgimg               |
| `motion_detect_dilation`        | `int`    | 10      | Iterations for dilation on fgimg              |
| `motion_detect_shadows`         | `bool`   | `false` | Whether shadows should be tracked             |
| `motion_detect_debug_contours`  | `bool`   | `false` | Whether contours should be tracked            |

## hector_tag_detection

The package for the `tag_detection` node

### tag_detection

The tag_detection node can detect tags of different types on images, read their content and store their position in the image.
Currently supported are QR-Codes and Apriltags of the 41h12 standart.

#### Subscribed Topics

| Topic                    | Type                     | Description                                    |
| ------------------------ | ------------------------ | ---------------------------------------------- |
| `/image`                 | `image_transport/camera` | The image that tags will be searched on        |
| `/tag_detection/enabled` | `std_msgs/msg/bool`      | The node can be enabled/disabled on this topic |

#### Published Topics

| Topic                           | Type                                 | Description                                                        |
| ------------------------------- | ------------------------------------ | ------------------------------------------------------------------ |
| `/perception/image_percept`     | `vision_msgs/msg/detection2_d_array` | An array of all tags detected on the input image                   |
| `/perception/debug`             | `sensor_msgs/msg/image`              | A cropped image for each detected tag                              |
| `/tag_detection/enabled_status` | `std_msgs/msg/bool`                  | The node will publish it's state here after being enabled/disabled |

#### Parameters

| Parameter         | Type     | Default                      | Description                            |
| ----------------- | -------- | ---------------------------- | -------------------------------------- |
| `enabled`         | `bool`   | `true`                       | Enables/disables the node              |
| `image_topic`     | `string` | `"image"`                    | Changes input topic for camera images  |
| `detection_topic` | `string` | `"perception/image_percept"` | Changes output topic for detected tags |
| `debug_topic`     | `string` | `"perception/debug"`         | Changes output topic for debug images  |

## hector_vision_test

A package for testing the other packages in hector_vision

### camera_dummy

This node is a simple standin for a camera, publishing a configurable series of images at a configurable frequency

#### Parameters

| Parameter         | Type     | Default                      | Description                                                   |
| ----------------- | -------- | ---------------------------- | ------------------------------------------------------------- |
| `image_dir`       | `string` | `"<share_directory>/images"` | Directory containing the images to be published               |
| `image_frequency` | `double` | `0.2`                        | Frequency with which images are changed                       |
| `image_frames`    | `int`    | `25`                         | Times each image is published before changing to the next one |
