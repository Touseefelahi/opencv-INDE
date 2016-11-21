__kernel void negaposi(
   __global uchar* src,
   int src_step, int src_offset,
   __global uchar* dst,
   int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   if (x >= dst_cols) return;
   int src_index = mad24(y, src_step, x + src_offset);
   int dst_index = mad24(y, dst_step, x + dst_offset);
   dst[dst_index] = 255 - src[src_index];
};