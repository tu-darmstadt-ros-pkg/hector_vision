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
| hector_motion_detection                                     | <span style="color:red">Not Ported</span> |
| hector_pipe_detection                                       | <span style="color:red">Not Ported</span> |
| hector_stair_detection                                      | <span style="color:red">Not Ported</span> |
| [hector_tag_detection](#hector_tag_detection)               | <span style="color:green">Ported</span>   |
| hector_thermal_image_conversion                             | <span style="color:red">Not Ported</span> |
| hector_thermal_self_filter                                  | <span style="color:red">Not Ported</span> |
| hector_vision_algorithm                                     | <span style="color:red">Not Ported</span> |
| hector_vision_algorithm_py                                  | <span style="color:red">Not Ported</span> |

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
| `/detection/visual_detection`   | `vision_msgs/msg/Detection2DArray` | Lists of perceptions that are aggragated       |
| `/detection_aggregator/enabled` | `std_msgs/msg/Bool`                | The node can be enabled/disabled on this topic |

#### Published Topics

| Topic                                    | Type                     | Description                                                             |
| ---------------------------------------- | ------------------------ | ----------------------------------------------------------------------- |
| `/detection/aggregated_detections_image` | `image_transport/Camera` | This image overlays the received detection on the received camera image |
| `/detection_aggregator/enabled_status`   | `std_msgs/msg/Bool`      | The node will publish it's state here after being enabled/disabled      |

#### Services

None

#### Actions

None

#### Parameters

| Parameter          | Type     | Description                                                                                                  |
| ------------------ | -------- | ------------------------------------------------------------------------------------------------------------ |
| `enabled`          | `bool`   | Enables/disables the node                                                                                    |
| `storage_duration` | `double` | Time after which detections are disregarded in seconds. Newer detections of the same objects replace the old |

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
| `/image/tag`                    | `image_transport/camera`             | A cropped image of each detected tag                               |
| `/tag_detection/enabled_status` | `std_msgs/msg/bool`                  | The node will publish it's state here after being enabled/disabled |

#### Services

None

#### Actions

None

#### Parameters

| Parameter | Type   | Description               |
| --------- | ------ | ------------------------- |
| `enabled` | `bool` | Enables/disables the node |
