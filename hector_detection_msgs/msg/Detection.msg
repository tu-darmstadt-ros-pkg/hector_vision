# The class of the detection e.g. for a manometer detection this will be 'manometer' whereas for a hazmat_detection, it
# will be 'hazmat_sign'
string object_class

# The object's support: TODO Work out how it is determined and special values
float64 support

# X-offset of sub image and contours of the detection in the original image
int32 x_offset

# Y-offset of sub image and contours of the detection in the original image
int32 y_offset

# (Optional) sub image where the detection took place e.g. when detecting a manometer value it may return the subimage
# that contains the manometer
sensor_msgs/Image sub_image

# (Optional) the contours of the detected object in the sub image
geometry_msgs/Polygon[] contours

# (Optional) The name of the detected object e.g. when detecting hazmat signs this field is used to name which hazmat
# sign was detected
string name

# (Optional) The data field is a general purpose field for string data. For example, it may be used in a qr code
# detection to return the content of the qr code
string data

# (Optional) The value field is a general purpose field for float data. For example, it may be used in a manometer
# detection to return the value displayed by the manometer
float64 value
