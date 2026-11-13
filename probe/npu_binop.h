/* probe/npu_binop.h — 通用二元逐元素算子提交器（E-108E，泛化自 probe/npuadd.c / E-106B-3）
 *
 * 与 probe/npuadd.c 的关系：
 *   npuadd.c 的 npu_add64() 把"算子无关"的提交逻辑（握手/分配/重定位/提交/同步）和
 *   "算子相关"的常量（命令流资产目录、三个旧 IOVA、64 元素、cmdbuf 顺序）写死在一起。
 *   本文件只做一件事：把那组常量变成参数（npu_binop_plan_t），另加一个
 *   `npu_binop(op, a, b, out)` 按算子名选 plan。
 *
 * 不改变任何已验证的行为：命令流仍然来自 ncc 编译产物，本库只负责
 * HANDSHAKE(2+5+4) → MEM ALLOC/mmap/MAP → 写内容 → 地址重定位 → 组装
 * mdw_cmd_in/subcmd_info/cmdbuf 描述符 → CMD RUN → 等待 + DMA_BUF_SYNC → 读回。
 * 不调用 libneuron / neuronrt。
 *
 * 证据状态（见 docs/OP-elementwise.md §7）：
 *   * ADD  : plan 用 E-106B-3 抓取的资产（/data/local/tmp/npuops/addin），与 npu_add64() 等价，已实测；
 *   * MUL / SUB / MAXIMUM / MINIMUM : **命令流资产尚未抓取**，npu_binop() 返回 NPU_BINOP_ENOASSET。
 *     抓取方式：用 probe/iolog2.c + probe/run_shim.sh 跑一次对应 .dla，把三个 cmdbuf、
 *     subcmd(80B)、cmdin(152B) 与三个旧 IOVA 落到资产目录即可（无需改本文件）。
 */
#ifndef NPU_BINOP_H
#define NPU_BINOP_H

#include <stdint.h>

/* op 名称（与 TFLite BuiltinOperator 同名，便于和 ops/<op>_<shape>_int8 对应） */
typedef enum {
  NPU_OP_ADD = 0,
  NPU_OP_MUL,
  NPU_OP_SUB,
  NPU_OP_MAXIMUM,
  NPU_OP_MINIMUM,
  NPU_OP_RELU,
  NPU_OP_LOGISTIC,
  NPU_OP_TANH,
  NPU_OP_FC_DYNW,
  NPU_OP_BMM_QK,
  NPU_OP_SOFTMAX16,
  NPU_OP_BMM_QK16,
  NPU_OP_BMM_PV16,
  NPU_OP__COUNT
} npu_op_t;

/* 返回值 */
#define NPU_BINOP_OK        0
#define NPU_BINOP_EINVAL   (-1)
#define NPU_BINOP_EIO      (-2)   /* 握手/分配/mmap 失败 */
#define NPU_BINOP_EASSET   (-3)   /* 资产文件读不到或大小不符 */
#define NPU_BINOP_EPATCH   (-4)   /* 命令流里找到的地址字个数与 expect_patches 不符 */
#define NPU_BINOP_ECALL    (-5)   /* CMD RUN 失败 */
#define NPU_BINOP_ENOASSET (-100) /* 该算子还没有抓取到的命令流资产 */

/* 一个算子的全部"算子相关"参数。数值全部来自一次合法运行（ncc 编译产物 + ioctl 抓取）。 */
typedef struct {
  const char *name;           /* 便于打印 */
  const char *asset_dir;      /* 设备上的资产目录 */
  uint64_t old_iova[3];       /* 抓取那次运行里 A、B、C 三个 buffer 的 IOVA */
  uint32_t bytes;             /* 每次提交处理的元素数（int8 => 字节数） */
  uint32_t bytes_b;           /* 第二个输入的字节数（FC 的动态权重是 4096）；0 => 同 bytes */
  uint32_t alloc_size;        /* A/B/C 的分配大小（抓取时 runtime 用 4 KiB，这里保持一致） */
  int cmdbuf_dir[3];          /* 三个 cmdbuf 的方向标志，顺序有语义（E-106B-3） */
  int n_inputs;               /* 1 = 一元算子(A→C)，2 = 二元算子(A,B→C) */
  int patch_idx;              /* 哪一个是含地址字的 cmdbuf（E-106B-3 实测是 368 B 那个） */
  int expect_patches;         /* 期望在命令流里找到的地址字个数（A/B/C 共 3 处） */
  const char *cmdbuf_file[3]; /* 资产文件名（按上面的语义顺序） */
} npu_binop_plan_t;

/* 提交一次 a OP b -> out（同形状 int8，无广播）。返回 NPU_BINOP_* 。
   npu_binop() 只是按算子选 plan；没有资产的算子返回 NPU_BINOP_ENOASSET 并在 stderr 说明。 */
int npu_bin_binop(const npu_binop_plan_t *plan, const int8_t *a, const int8_t *b, int8_t *out);
int npu_bin_unop(const npu_binop_plan_t *plan, const int8_t *a, int8_t *out);
int npu_binop(npu_op_t op, const int8_t *a, const int8_t *b, int8_t *out);

/* 返回某个算子的 plan（NULL = 还没抓取命令流资产）。 */
const npu_binop_plan_t *npu_binop_plan(npu_op_t op);

/* 元素数（int8）：等于 plan->bytes。 */
uint32_t npu_binop_elems(npu_op_t op);

#endif /* NPU_BINOP_H */
