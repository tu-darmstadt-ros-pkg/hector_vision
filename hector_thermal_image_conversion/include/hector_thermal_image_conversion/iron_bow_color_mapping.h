#ifndef HECTOR_THERMAL_IMAGE_CONVERSION_IRON_BOW_COLOR_MAPPING_H_
#define HECTOR_THERMAL_IMAGE_CONVERSION_IRON_BOW_COLOR_MAPPING_H_

#include <cv_bridge/cv_bridge.h>

static std::vector<cv::Vec3b> iron_bow_color_mapping = {cv::Vec3b(0, 0, 19), cv::Vec3b(0, 0, 37), cv::Vec3b(0, 0, 42),
                                                 cv::Vec3b(0, 0, 50), cv::Vec3b(0, 0, 58), cv::Vec3b(0, 0, 62),
                                                 cv::Vec3b(0, 0, 70), cv::Vec3b(0, 0, 79), cv::Vec3b(1, 0, 85),
                                                 cv::Vec3b(1, 0, 87), cv::Vec3b(2, 0, 92), cv::Vec3b(4, 0, 97),
                                                 cv::Vec3b(4, 0, 99), cv::Vec3b(6, 0, 104), cv::Vec3b(8, 0, 108),
                                                 cv::Vec3b(9, 0, 110), cv::Vec3b(11, 0, 115), cv::Vec3b(13, 0, 117),
                                                 cv::Vec3b(13, 0, 118), cv::Vec3b(16, 0, 121), cv::Vec3b(20, 0, 123),
                                                 cv::Vec3b(22, 0, 125), cv::Vec3b(25, 0, 127), cv::Vec3b(29, 0, 130),
                                                 cv::Vec3b(31, 0, 132), cv::Vec3b(34, 0, 134), cv::Vec3b(38, 0, 136),
                                                 cv::Vec3b(42, 0, 138), cv::Vec3b(44, 0, 139), cv::Vec3b(49, 0, 141),
                                                 cv::Vec3b(52, 0, 142), cv::Vec3b(54, 0, 143), cv::Vec3b(57, 0, 144),
                                                 cv::Vec3b(60, 0, 146), cv::Vec3b(62, 0, 147), cv::Vec3b(65, 0, 148),
                                                 cv::Vec3b(68, 0, 149), cv::Vec3b(70, 0, 150), cv::Vec3b(73, 0, 150),
                                                 cv::Vec3b(77, 0, 151), cv::Vec3b(80, 0, 151), cv::Vec3b(82, 0, 151),
                                                 cv::Vec3b(85, 0, 152), cv::Vec3b(89, 0, 153), cv::Vec3b(91, 0, 154),
                                                 cv::Vec3b(94, 0, 154), cv::Vec3b(98, 0, 155), cv::Vec3b(99, 0, 155),
                                                 cv::Vec3b(103, 0, 155), cv::Vec3b(106, 0, 155), cv::Vec3b(108, 0, 156),
                                                 cv::Vec3b(112, 0, 156), cv::Vec3b(114, 0, 157), cv::Vec3b(116, 0, 157),
                                                 cv::Vec3b(120, 0, 157), cv::Vec3b(123, 0, 157), cv::Vec3b(125, 0, 157),
                                                 cv::Vec3b(127, 0, 157), cv::Vec3b(131, 0, 157), cv::Vec3b(135, 0, 157),
                                                 cv::Vec3b(136, 0, 157), cv::Vec3b(139, 0, 156), cv::Vec3b(142, 0, 156),
                                                 cv::Vec3b(144, 0, 156), cv::Vec3b(147, 0, 156), cv::Vec3b(150, 0, 155),
                                                 cv::Vec3b(152, 0, 155), cv::Vec3b(155, 0, 155), cv::Vec3b(158, 0, 155),
                                                 cv::Vec3b(160, 0, 155), cv::Vec3b(163, 0, 154), cv::Vec3b(165, 0, 154),
                                                 cv::Vec3b(167, 0, 154), cv::Vec3b(168, 0, 154), cv::Vec3b(171, 0, 153),
                                                 cv::Vec3b(174, 0, 152), cv::Vec3b(175, 1, 152), cv::Vec3b(177, 1, 151),
                                                 cv::Vec3b(178, 2, 150), cv::Vec3b(179, 2, 150), cv::Vec3b(181, 2, 149),
                                                 cv::Vec3b(182, 2, 149), cv::Vec3b(183, 3, 149), cv::Vec3b(186, 4, 148),
                                                 cv::Vec3b(187, 4, 147), cv::Vec3b(188, 5, 146), cv::Vec3b(190, 5, 146),
                                                 cv::Vec3b(191, 6, 145), cv::Vec3b(191, 6, 145), cv::Vec3b(193, 8, 144),
                                                 cv::Vec3b(194, 10, 143), cv::Vec3b(196, 11, 141), cv::Vec3b(196, 12, 140),
                                                 cv::Vec3b(198, 13, 138), cv::Vec3b(199, 14, 137), cv::Vec3b(200, 15, 136),
                                                 cv::Vec3b(201, 17, 134), cv::Vec3b(202, 19, 132), cv::Vec3b(203, 19, 131),
                                                 cv::Vec3b(204, 21, 129), cv::Vec3b(206, 23, 125), cv::Vec3b(206, 24, 124),
                                                 cv::Vec3b(207, 25, 121), cv::Vec3b(209, 27, 118), cv::Vec3b(210, 29, 115),
                                                 cv::Vec3b(211, 30, 114), cv::Vec3b(212, 32, 111), cv::Vec3b(213, 35, 108),
                                                 cv::Vec3b(214, 36, 106), cv::Vec3b(215, 38, 102), cv::Vec3b(217, 40, 98),
                                                 cv::Vec3b(217, 41, 96), cv::Vec3b(218, 43, 92), cv::Vec3b(219, 46, 88),
                                                 cv::Vec3b(219, 47, 85), cv::Vec3b(221, 49, 78), cv::Vec3b(222, 51, 71),
                                                 cv::Vec3b(222, 52, 68), cv::Vec3b(223, 54, 62), cv::Vec3b(224, 57, 55),
                                                 cv::Vec3b(225, 58, 52), cv::Vec3b(226, 59, 45), cv::Vec3b(227, 61, 39),
                                                 cv::Vec3b(228, 63, 33), cv::Vec3b(228, 64, 30), cv::Vec3b(229, 67, 25),
                                                 cv::Vec3b(230, 69, 23), cv::Vec3b(230, 70, 22), cv::Vec3b(231, 72, 19),
                                                 cv::Vec3b(231, 74, 17), cv::Vec3b(232, 75, 16), cv::Vec3b(232, 76, 14),
                                                 cv::Vec3b(233, 78, 12), cv::Vec3b(234, 79, 11), cv::Vec3b(235, 81, 10),
                                                 cv::Vec3b(236, 84, 9), cv::Vec3b(236, 86, 8), cv::Vec3b(236, 87, 7),
                                                 cv::Vec3b(236, 89, 6), cv::Vec3b(237, 91, 5), cv::Vec3b(237, 92, 5),
                                                 cv::Vec3b(238, 93, 4), cv::Vec3b(238, 95, 4), cv::Vec3b(239, 96, 4),
                                                 cv::Vec3b(239, 99, 4), cv::Vec3b(240, 100, 3), cv::Vec3b(240, 101, 3),
                                                 cv::Vec3b(241, 103, 2), cv::Vec3b(241, 104, 2), cv::Vec3b(241, 105, 2),
                                                 cv::Vec3b(242, 108, 1), cv::Vec3b(242, 109, 1), cv::Vec3b(243, 110, 1),
                                                 cv::Vec3b(243, 112, 1), cv::Vec3b(244, 114, 0), cv::Vec3b(244, 115, 0),
                                                 cv::Vec3b(244, 116, 0), cv::Vec3b(244, 118, 0), cv::Vec3b(244, 120, 0),
                                                 cv::Vec3b(244, 122, 0), cv::Vec3b(245, 124, 0), cv::Vec3b(245, 126, 0),
                                                 cv::Vec3b(245, 128, 0), cv::Vec3b(246, 130, 0), cv::Vec3b(247, 132, 0),
                                                 cv::Vec3b(247, 133, 0), cv::Vec3b(247, 135, 0), cv::Vec3b(248, 137, 0),
                                                 cv::Vec3b(248, 138, 0), cv::Vec3b(248, 139, 0), cv::Vec3b(248, 141, 0),
                                                 cv::Vec3b(249, 142, 0), cv::Vec3b(249, 143, 0), cv::Vec3b(250, 145, 0),
                                                 cv::Vec3b(250, 148, 0), cv::Vec3b(250, 149, 0), cv::Vec3b(250, 151, 0),
                                                 cv::Vec3b(251, 153, 0), cv::Vec3b(251, 154, 0), cv::Vec3b(251, 157, 0),
                                                 cv::Vec3b(252, 160, 0), cv::Vec3b(252, 161, 0), cv::Vec3b(253, 164, 0),
                                                 cv::Vec3b(253, 167, 0), cv::Vec3b(253, 168, 0), cv::Vec3b(253, 169, 0),
                                                 cv::Vec3b(253, 172, 0), cv::Vec3b(253, 174, 0), cv::Vec3b(253, 175, 0),
                                                 cv::Vec3b(254, 177, 0), cv::Vec3b(254, 179, 0), cv::Vec3b(254, 180, 0),
                                                 cv::Vec3b(254, 182, 0), cv::Vec3b(254, 184, 0), cv::Vec3b(254, 185, 0),
                                                 cv::Vec3b(254, 187, 0), cv::Vec3b(254, 190, 0), cv::Vec3b(254, 191, 0),
                                                 cv::Vec3b(254, 193, 0), cv::Vec3b(254, 196, 0), cv::Vec3b(254, 198, 0),
                                                 cv::Vec3b(254, 199, 1), cv::Vec3b(254, 200, 1), cv::Vec3b(254, 202, 2),
                                                 cv::Vec3b(254, 203, 2), cv::Vec3b(254, 205, 2), cv::Vec3b(254, 207, 3),
                                                 cv::Vec3b(254, 208, 4), cv::Vec3b(254, 209, 5), cv::Vec3b(254, 211, 7),
                                                 cv::Vec3b(254, 212, 8), cv::Vec3b(254, 215, 11), cv::Vec3b(254, 216, 12),
                                                 cv::Vec3b(254, 217, 13), cv::Vec3b(255, 219, 16), cv::Vec3b(255, 220, 19),
                                                 cv::Vec3b(255, 221, 21), cv::Vec3b(255, 222, 25), cv::Vec3b(255, 223, 30),
                                                 cv::Vec3b(255, 225, 34), cv::Vec3b(255, 226, 36), cv::Vec3b(255, 227, 41),
                                                 cv::Vec3b(255, 229, 46), cv::Vec3b(255, 229, 50), cv::Vec3b(255, 231, 57),
                                                 cv::Vec3b(255, 233, 64), cv::Vec3b(255, 234, 68), cv::Vec3b(255, 235, 75),
                                                 cv::Vec3b(255, 236, 81), cv::Vec3b(255, 237, 85),cv::Vec3b(255, 238, 92),
                                                 cv::Vec3b(255, 239, 99),cv::Vec3b(255, 240, 107), cv::Vec3b(255, 241, 111),
                                                 cv::Vec3b(255, 242, 121), cv::Vec3b(255, 242, 130), cv::Vec3b(255, 243, 135),
                                                 cv::Vec3b(255, 243, 144), cv::Vec3b(255, 244, 152), cv::Vec3b(255, 244, 156),
                                                 cv::Vec3b(255, 245, 164), cv::Vec3b(255, 246, 172), cv::Vec3b(255, 247, 176),
                                                 cv::Vec3b(255, 248, 184), cv::Vec3b(255, 249, 191), cv::Vec3b(255, 249, 194),
                                                 cv::Vec3b(255, 249, 201), cv::Vec3b(255, 250, 208), cv::Vec3b(255, 250, 211),
                                                 cv::Vec3b(255, 251, 217), cv::Vec3b(255, 252, 224), cv::Vec3b(255, 253, 231),
                                                 cv::Vec3b(255, 253, 234), cv::Vec3b(255, 254, 239), cv::Vec3b(255, 255, 245),
                                                 cv::Vec3b(255, 255, 248)};

#endif //HECTOR_THERMAL_IMAGE_CONVERSION_IRON_BOW_COLOR_MAPPING_H_
