/*
JPEG decoding with libjpeg(-turbo) DCT scaling: decoding a 12 MP MetaCam frame at 1/4 resolution skips most of the
inverse DCT work, which cv::imdecode does not offer (it ignores IMREAD_REDUCED_* in OpenCV 4.2).
*/

#ifndef JPEG_DECODE_H_
#define JPEG_DECODE_H_

#include <csetjmp>
#include <cstdio>
#include <jpeglib.h>
#include <opencv2/core.hpp>

namespace jpeg_decode
{

struct ErrorManager
{
  jpeg_error_mgr pub;
  jmp_buf jump;
};

inline void errorExit(j_common_ptr cinfo) { longjmp(reinterpret_cast<ErrorManager *>(cinfo->err)->jump, 1); }

/// Decode a JPEG into BGR8, downscaled by `denom` (1, 2, 4 or 8). Returns an empty Mat on failure.
inline cv::Mat decodeBGR(const unsigned char *data, size_t size, int denom)
{
  jpeg_decompress_struct cinfo;
  ErrorManager jerr;
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = errorExit;
  cv::Mat *out = new cv::Mat();
  if (setjmp(jerr.jump))
  {
    jpeg_destroy_decompress(&cinfo);
    delete out;
    return cv::Mat();
  }
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, const_cast<unsigned char *>(data), size);
  jpeg_read_header(&cinfo, TRUE);
  cinfo.scale_num = 1;
  cinfo.scale_denom = denom;
  cinfo.out_color_space = JCS_EXT_BGR;
  jpeg_start_decompress(&cinfo);
  out->create(cinfo.output_height, cinfo.output_width, CV_8UC3);
  while (cinfo.output_scanline < cinfo.output_height)
  {
    JSAMPROW row = out->ptr<unsigned char>(cinfo.output_scanline);
    jpeg_read_scanlines(&cinfo, &row, 1);
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  cv::Mat result = *out;
  delete out;
  return result;
}

} // namespace jpeg_decode

#endif // JPEG_DECODE_H_
