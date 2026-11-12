/* 自建 NPU ADD 函数库（E-106B-3）
 * 固定形状：64 个 int8 元素逐位相加；命令流来自 ncc 编译产物，本库负责资源分配、
   地址重定位、提交与完成等待——不调用 libneuron / neuronrt。返回 0 成功。 */
#ifndef NPUADD_H
#define NPUADD_H
#include <stdint.h>
int npu_add64(const int8_t *a, const int8_t *b, int8_t *out);
#endif
