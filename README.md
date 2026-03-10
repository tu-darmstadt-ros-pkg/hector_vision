# Hector Vision

## Status

| Package                                                     | Status                                    |
| ----------------------------------------------------------- | ----------------------------------------- |
| hector_bar_detection                                        | <span style="color:red">Not Ported</span> |
| hector_barrel_detection_nodelet                             | <span style="color:red">Not Ported</span> |
| hector_barrels_detection                                    | <span style="color:red">Not Ported</span> |
| hector_color_detection_nodelet                              | <span style="color:red">Not Ported</span> |
| [hector_detection_aggregator](#hector_detection_aggregator) | <span style="color:green">Ported</span>   |
| hector_head_detection                                       | <span style="color:red">Not Ported</span> |
| hector_image_rotate                                         | <span style="color:red">Not Ported</span> |
| [hector_motion_detection](#hector_motion_detection)         | <span style="color:green">Ported</span>   |
| hector_pipe_detection                                       | <span style="color:red">Not Ported</span> |
| hector_stair_detection                                      | <span style="color:red">Not Ported</span> |
| [hector_tag_detection](#hector_tag_detection)               | <span style="color:green">Ported</span>   |
| hector_thermal_image_conversion                             | <span style="color:red">Not Ported</span> |
| hector_thermal_self_filter                                  | <span style="color:red">Not Ported</span> |
| hector_vision_algorithm                                     | <span style="color:red">Not Ported</span> |
| hector_vision_algorithm_py                                  | <span style="color:red">Not Ported</span> |
| [hector_vision_test](#hector_vision_test)                   | <span style="color:blue">New</span>       |

## hector_detection_aggregator

The package for the `detection_aggregator` node

### detection_aggregator

This node collects data on detected objects and overlays them on an image feed.
Additionally detected contacts are buffered for a configurable ammount of time.
The detections are tagged with the detection type and an additional value (e.g. the string of a qr-code).
Detections are colored according to their type:

| Detection Type | Color     |
| -------------- | --------- |
| motion         | red       |
| qr-code        | blue      |
| apriltag       | purple    |
| heat           | green     |
| hazmat         | turquoise |
| unknown/other  | grey      |

#### Subscribed Topics

| Topic                           | Type                               | Description                                    |
| ------------------------------- | ---------------------------------- | ---------------------------------------------- |
| `/image`                        | `image_transport/Camera`           | The image that detections are overlaid on      |
| `/detection/visual_detection`   | `vision_msgs/msg/Detection2DArray` | Lists of perceptions that are aggregated       |
| `/detection_aggregator/enabled` | `std_msgs/msg/Bool`                | The node can be enabled/disabled on this topic |

#### Published Topics

| Topic                                    | Type                     | Description                                                             |
| ---------------------------------------- | ------------------------ | ----------------------------------------------------------------------- |
| `/detection/aggregated_detections_image` | `image_transport/Camera` | This image overlays the received detection on the received camera image |
| `/detection_aggregator/enabled_status`   | `std_msgs/msg/Bool`      | The node will publish it's state here after being enabled/disabled      |

#### Parameters

| Parameter           | Type     | Default                                   | Description                                                                                                                                               |
| ------------------- | -------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`           | `bool`   | `true`                                    | Enables/disables the node                                                                                                                                 |
| `detection_topic`   | `string` | `"detection/visual_detection"`            | Parameter for setting the detection topic                                                                                                                 |
| `aggregation_topic` | `string` | `"detection/aggregated_detections_image"` | Parameter for setting the aggregation topic                                                                                                               |
| `aggregation_mode`  | `string` | `"NEWEST"`                                | Determines how detections and immages are aggregated. Possible values are `"NEWEST"` or `"COMPLETE"`                                                      |
| `storage_duration`  | `double` | `1.5`                                     | Time after which detections are disregarded in seconds. Newer detections of the same objects replace the old. Only used if `aggregation_mode` is `NEWEST` |
| `buffer_size`       | `int`    | `16`                                      | Number of frames (both detections and images) that can be buffered. Only used when `aggregation_mode` is `COMPLETE`                                       |
| `robot_namespace`   | `string` | `""`                                      | Parameter to enforce namespacing of topics                                                                                                                |

## hector_motion_detection

The package for the `motion_detection` node

### motion_detection

The motion_detection node can detect multiple moving objects in an image sequence.

#### Subscribed Topics

| Topic     | Type                     | Description                             |
| --------- | ------------------------ | --------------------------------------- |
| `/enable` | `std_msgs/msg/bool`      | Enables or disables the node            |
| `/image`  | `image_transport/camera` | The image that tags will be searched on |

#### Published Topics

| Topic                         | Type                                 | Description                 |
| ----------------------------- | ------------------------------------ | --------------------------- |
| `enabled_status`              | `std_msgs/msg/bool`                  | Whether the node is enabled |
| `detection/image_detection`   | `vision_msgs/msg/detection2_d_array` | The detected moving objects |
| `image_motion`                | `sensor_msgs/msg/image`              | TODO                        |
| `image_detected`              | `sensor_msgs/msg/image`              | TODO                        |
| `image_background_subtracted` | `sensor_msgs/msg/image`              | TODO                        |

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

#### Published Topics

| Topic          | Type                         | Description                                   |
| -------------- | ---------------------------- | --------------------------------------------- |
| `/image`       | `sensor_msgs/msg/image`      | The image sequence is published on this topic |
| `/camera_info` | `sensor_msgs/msg/CameraInfo` | Camera info is currently left empty           |

#### Parameters

| Parameter         | Type     | Default                      | Description                                                   |
| ----------------- | -------- | ---------------------------- | ------------------------------------------------------------- |
| `image_dir`       | `string` | `"<share_directory>/images"` | Directory containing the images to be published               |
| `image_frequency` | `double` | `0.2`                        | Frequency with which images are changed                       |
| `image_frames`    | `int`    | `25`                         | Times each image is published before changing to the next one |

